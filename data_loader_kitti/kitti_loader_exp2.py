import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
import pathlib

import open3d as o3d


kitti_cache = {}
kitti_icp_cache = {}


def voxel_downsample(xyz, voxel_size=0.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    down_pcd = pcd.voxel_down_sample(voxel_size=0.5)
    down_pcd_np = np.asarray(down_pcd.points)

    return down_pcd_np

# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def sample_random_trans(pcd, randg, rotation_range=0):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


class PairDataset(torch.utils.data.Dataset):
    AUGMENT = None

    def __init__(self, phase, transform=None, random_rotation=True, random_scale=True,
                    manual_seed=False, config=None):

        self.phase = phase 
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = 0.5
        self.matching_search_voxel_size = 0.5 * 1.5

        self.random_scale = random_scale
        self.random_rotation = random_rotation
        self.rotation_range = 360
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)
    
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts
    
    def __len__(self):
        return len(self.files)


class KITTIPairDataset(PairDataset):
    AUGMENT = None 
    DATA_FILES = {'train': 'C:/Users/praop/OneDrive/Desktop/NYU/AI4CE/code/DeepMapping++/data_loader_kitti/config/train_kitti.txt',
                'val': 'C:/Users/praop/OneDrive/Desktop/NYU/AI4CE/code/DeepMapping++/data_loader_kitti/config/val_kitti.txt',
                'test': 'C:/Users/praop/OneDrive/Desktop/NYU/AI4CE/code/DeepMapping++/data_loader_kitti/config/train_kitti.txt'}

    #DATA_FILES = {'train': '/scratch/pr2257/ai4ce/DeepMapping++/data_loader_kitti/config/train_kitti.txt',
    #            'val': '/scratch/pr2257/ai4ce/DeepMapping++/data_loader_kitti/config/val_kitti.txt',
    #            'test': '/scratch/pr2257/ai4ce/DeepMapping++/data_loader_kitti/config/test_kitti.txt'}
    TEST_RANDOM_ROTATION = True
    IS_ODOMETRY = True 


    def __init__(self, phase='train', transform=None, random_rotation=True, random_scale=False, manual_seed=False, config=None):

        if self.IS_ODOMETRY:
            self.root = root = 'D:\KITTI_odom\dataset_velodyne'
            #self.root = root = '/scratch/pr2257/ai4ce/data/KITTI_odom/dataset_velodyne'

            random_rotation = self.TEST_RANDOM_ROTATION

        self.icp_path = os.path.join('D:\KITTI_odom\dataset_velodyne', 'icp_2')
        #self.icp_path = os.path.join('/scratch/pr2257/ai4ce/data/KITTI_odom/dataset_velodyne', 'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        PairDataset.__init__(self, phase, transform, random_rotation, random_scale, manual_seed, config)

        logging.info(f"Loading the subset {phase} from {root}")

        # Use the kitti root
        self.max_time_diff = max_time_diff = 3

        subset_names = open(self.DATA_FILES[phase]).read().split()

        #self.min_points = self.get_mininum_points(subset_names)

        #print(self.min_points)

        # 4415
        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
            for start_time in inames:
                for time_diff in range(2, max_time_diff):
                    pair_time = time_diff + start_time
                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))



    def get_mininum_points(self, subset_names):

        pcd = o3d.geometry.PointCloud()


        min_points = []
        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
             
            drive_min_points = 0

            for name in inames:
                fname = self._get_velodyne_fn(drive_id, name)
                point_cloud = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
                point_cloud = point_cloud[:, 0:3] #<N, x, y, z>
                point_cloud = np.asarray(point_cloud, np.float32)

                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                down_pcd = pcd.voxel_down_sample(voxel_size=0.5)
                down_pcd_np = np.asarray(down_pcd.points)

                if drive_min_points == 0 or drive_min_points > down_pcd_np.shape[0]:
                    drive_min_points = down_pcd_np.shape[0]

            min_points.append(drive_min_points)
            

        return min(min_points)
       

    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                        '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam
    
    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in kitti_cache:
                kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return kitti_cache[data_path]
            else:
                return kitti_cache[data_path][indices]
        else:
            data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(self.root + '/' + self.date + '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in kitti_cache:
                    kitti_cache[filename] = np.genfromtxt(filename)
                    odometry.append(kitti_cache[filename])

            odometry = np.array(odometry)
            return odometry
    
    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    
    def __getitem__(self, idx):
        
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)


        pcd = o3d.geometry.PointCloud()

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):

                #M = (self.velo2cam() @ positions[0].T @ np.linalg.inv(positions[1].T) @ np.linalg.inv(self.velo2cam())).T
                #M = np.linalg.inv(np.linalg.inv(self.velo2cam()) @ np.linalg.inv(positions[0]) @ positions[1] @ self.velo2cam())
                
                M = (np.linalg.inv(self.velo2cam()) @ np.linalg.inv(positions[1]) @ positions[0] @ self.velo2cam())

                down_pcd0 = voxel_downsample(xyz0, 0.05)
                down_pcd1 = voxel_downsample(xyz1, 0.05)

                down_pcd0_t = self.apply_transform(down_pcd0, M)
                pcd0 = make_open3d_point_cloud(down_pcd0_t)
                pcd1 = make_open3d_point_cloud(down_pcd1)


                reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4), 
                o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

                pcd0.transform(reg.transformation)
                #M2 = M @ reg.transformation
                M2 = M
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
            T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)

        else:
            trans = M2

        # Voxelization

        pcd0 = voxel_downsample(xyz0, self.voxel_size)
        pcd1 = voxel_downsample(xyz1, self.voxel_size)
        
        pcd0 = pcd0[:4415, :]
        pcd1 =  pcd1[:4415, :]

        pcd0 = pcd0[::3, :]
        pcd1 = pcd1[::3, :]

        pc_pair = [pcd0, pcd1]

        #pc_pair = [xyz0, xyz1]
        pc_pair = np.asarray(pc_pair)

        pc_pair_th = torch.from_numpy(pc_pair).float()
        pose = torch.zeros(1,4,dtype=torch.float32)

        return pc_pair_th, pose, trans

class KITTINMPairDataset(KITTIPairDataset):

    MIN_DIST = 2

    def __init__(self, phase, transform=None, random_rotation=True, random_scale=True,manual_seed=False, config=None):

        if self.IS_ODOMETRY:
            self.root = root = 'D:\KITTI_odom\dataset_velodyne'
            #self.root = root = '/scratch/pr2257/ai4ce/data/KITTI_odom/dataset_velodyne'
            random_rotation = self.TEST_RANDOM_ROTATION

        self.icp_path = os.path.join('D:\KITTI_odom\dataset_velodyne', 'icp_2')
        #self.icp_path = os.path.join('/scratch/pr2257/ai4ce/data/KITTI_odom/dataset_velodyne', 'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        PairDataset.__init__(self, phase, transform, random_rotation, random_scale, manual_seed, config)

        logging.info(f"Loading the subset {phase} from {root}")

        subset_names = open(self.DATA_FILES[phase]).read().split()

        if self.IS_ODOMETRY:
            for dirname in subset_names:
                drive_id = int(dirname)
                fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)
                assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, :3, 3]
                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
                pdist = np.sqrt(pdist.sum(-1))
                valid_pairs = pdist > self.MIN_DIST
                curr_time = inames[0]
                while curr_time in inames:
                    # Find the min index
                    next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
                    if len(next_time) == 0:
                        curr_time += 1
                    else:
                        # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                        next_time = next_time[0] + curr_time - 1

                    if next_time in inames:
                        self.files.append((drive_id, curr_time, next_time))
                        curr_time = next_time + 1

                    if self.IS_ODOMETRY:
                        # Remove problematic sequence
                        for item in [(8, 15, 58)]:
                            if item in self.files:
                                self.files.pop(self.files.index(item))
        self.files = self.files[0:100]

#dataset = KITTINMPairDataset('train')
#pc_pair, pose = dataset.__getitem__(3)

#print(pc_pair.shape)
#print(len(dataset.files))
