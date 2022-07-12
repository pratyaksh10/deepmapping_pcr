import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import open3d as o3d
from tqdm import tqdm

def find_valid_points(local_point_cloud):
    """
    find valid points in local point cloud
        invalid points have all zeros local coordinates
    local_point_cloud: <BxNxk> 
    valid_points: <BxN> indices  of valid point (0/1)
    """
    eps = 1e-6
    non_zero_coord = torch.abs(local_point_cloud) > eps
    valid_points = torch.sum(non_zero_coord, dim=-1)
    valid_points = valid_points > 0

    return valid_points

class Kitti(Dataset):
    def __init__(self, root, traj, voxel_size=1, init_pose=None,
                group=True, group_size=8, pairwise=False, use_tqdm=True, **kwargs):
        self.radius = 6378137 # earth radius
        self.root = root
        self.traj = traj
        data_folder = os.path.join(root, traj)
        self.init_pose=init_pose
        self.group_flag = group
        self.pairwise_flag = pairwise
        if self.pairwise_flag:
            self.pairwise_pose = kwargs["pairwise_pose"]
        if self.pairwise_flag and not self.group_flag:
            print("Pairwise registration needs group information")
            assert()

        files = os.listdir(data_folder)
        files.remove('gt_pose.npy')
        files.remove('gt_trans.npy')
        try:
            files.remove('group_matrix.npy')
        except:
            pass
        point_clouds = []
        max_points = 0
        
        path = r'D:\kitti_group\2011_09_30_drive_0018_sync_tfvpr\gt_pose.npy'

        # point_clouds = np.load(os.path.join(data_folder, 'point_cloud.npy')).astype('float32')
        #gt_pose = np.load(os.path.join(data_folder, 'gt_pose.npy')).astype('float32')
        gt_pose = np.load(path)
        gt_pose[:, :2] *= np.pi / 180
        lat_0 = gt_pose[0, 0]
        gt_pose[:, 1] *= self.radius * np.cos(lat_0)
        gt_pose[:, 0] *= self.radius
        gt_pose[:, 1] -= gt_pose[0, 1]
        gt_pose[:, 0] -= gt_pose[0, 0]

        self.gt_pose = gt_pose[:, [1, 0, 2, 3, 4, 5]] #<x,y,z,r,p,y> Kx6
    
        
        for file in tqdm(files, disable=not use_tqdm):
            # xyz = np.load(os.path.join(data_folder, file))
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # _, inliers = pcd.segment_plane(distance_threshold=0.1,
            #                              ransac_n=10,
            #                              num_iterations=1000)
            # pcd.select_by_index(inliers, invert=True)
            pcd = o3d.io.read_point_cloud(os.path.join(data_folder, file))
            pcd = pcd.voxel_down_sample(voxel_size)
            pcd = np.asarray(pcd.points)
            pcd = pcd[:, :]
            point_clouds.append(pcd)
            if max_points < pcd.shape[0]:
                max_points = pcd.shape[0]
        # print(max_points)
        for i in range(len(point_clouds)):
            point_clouds[i] = np.pad(point_clouds[i], ((0, max_points-point_clouds[i].shape[0]), (0, 0)))
        
        
        
        self.point_clouds = torch.from_numpy(np.stack(point_clouds)).float() # <BxNx3>
        print('shape of point cloud:', self.point_clouds.shape)
        self.n_pc = self.point_clouds.shape[0]
        self.n_points = self.point_clouds.shape[1]
        #self.valid_points = find_valid_points(self.point_clouds)
        #max_dst = utils.transform_to_global_KITTI(torch.tensor(self.gt_pose), self.point_clouds).max().item()
        #self.point_clouds /= max_dst
        #self.init_pose[:, :2] = self.init_pose[:, :2] / max_dst
        #self.gt_pose[:, :2] =  self.gt_pose[:, :2] / max_dst
        #if self.group_flag:
        #    self.group_matrix = np.load(os.path.join(data_folder, 'group_matrix.npy')).astype('int')
        #    if self.group_matrix.shape[1] < group_size:
        #        print("Warning: matrix size {} is smaller than group size {}, using {}".format(self.group_matrix.shape[1], kwargs['group_size'], self.group_matrix.shape[1]))
        #    else:
        #        self.group_matrix = self.group_matrix[:, :group_size]

        pair_pcs, pair_gt_trans = self.generate_pairs(self.point_clouds, self.group_matrix)
        np.save(r'/mnt/NAS/home/xinhao/pcr_prat/deepmapping_pcr/group_gt_trans.npy', pair_gt_trans)
        np.save(r'/mnt/NAS/home/xinhao/pcr_prat/deepmapping_pcr/pc_pairs_01.npy', pair_pcs)
        

        #pair_gt_trans = np.load(r'/mnt/NAS/home/xinhao/pcr_prat/deepmapping_pcr/group_gt_trans.npy')

        #pair_pcs = np.load(r'/mnt/NAS/home/xinhao/pcr_prat/deepmapping_pcr/pc_pairs_01.npy')

        
        #pair_pcs = np.load(r'/mnt/NAS/home/xinhao/pcr_prat/group_pairs.npy')
        #self.pair_pcs = pair_pcs[:50, :, :, :]
        #self.gt_trans = pair_gt_trans[:50, :, :]
        self.n_pc = self.pair_pcs.shape[0]

        self.N = self.pair_pcs.shape[2]
        
        #np.save(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\group_pairs.npy', pair_pcs.cpu().detach().numpy())

    def rot3d(self, axis, angle):
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(3)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

    def pos_transform(self, pos):
        x, y, z, rx, ry, rz = pos
        RT = np.eye(4)
        RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
        RT[:3, 3] = [x, y, z]
        return RT
    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)) if not invert else np.dot(
            np.linalg.inv(T1), T0))

    def __getitem__(self,index):

        '''
        if self.group_flag:
            G = self.group_matrix.shape[1]
            indices = self.group_matrix[index]
            pcd = self.point_clouds[indices, :, :]  # <GxNx3>

            centre_frame = pcd[0, :, :].unsqueeze(0)      # <1xNx3>

            centre_frame = centre_frame.repeat(G-1, 1, 1).unsqueeze(1) #<G-1x1xNx3>

            spatial_neigbors = pcd[1:, :, :].unsqueeze(1) # <G-1x1xNx3>

            pairs = torch.cat((spatial_neigbors, centre_frame), 1) #<G-1x2xNx3>            
            valid_points = self.valid_points[indices,:]  # <GxN>

            if self.init_pose is not None:
                # pcd = pcd.unsqueeze(0)  # <1XNx3>
                init_global_pose = self.init_pose[indices, :] # <Gx4>
                # pcd = utils.transform_to_global_KITTI(pose, pcd).squeeze(0)
            else:
                init_global_pose = torch.zeros(self.group_matrix.shape[1], 4)
            if self.pairwise_flag:
                pairwise_pose = self.pairwise_pose[index]
                pairwise_pose = torch.tensor(pairwise_pose)
            else:
                pairwise_pose = torch.zeros(indices.shape[0]-1, 4)
            return pairs                     # <G-1x2xNx3>
        else:
            return self.point_clouds[index]
        '''

        pcd_pair = self.pair_pcs[index,:,:, :] #<2xNx3>, <source_pc, template_pc>
        sample_idx = np.random.choice(self.N, 1024, replace=False)
        pcd0 = pcd_pair[0, :, :]
        pcd1 = pcd_pair[1, :, :]
        pcd0 = pcd0[sample_idx, :]
        pcd1 = pcd1[sample_idx, :]
        Tr = self.gt_trans[index, :, :] # <4, 4>

        min_v1, max_v1 = np.min(pcd0),np.max(pcd0)
        s1 = (max_v1 - min_v1)

        min_v2, max_v2 = np.min(pcd1),np.max(pcd1)
        s2 = (max_v2 - min_v2)

        s = np.max([s1,s2])

        pcd0 = (pcd0 - min_v1)/s
        pcd1 = (pcd1 - min_v2)/s

        lala = np.ones_like(Tr[:3,3])*min_v1
        translations = (Tr[:3,3] - min_v2 + lala@ Tr[:3,:3].T)/s
        pcd1 = pcd1 - translations + Tr[:3,3]

        pc_pair = [pcd0, pcd1]

        #pc_pair = [xyz0, xyz1]
        pc_pair = np.asarray(pc_pair)

        pc_pair_th = torch.from_numpy(pc_pair).float()
  
        
        pose = torch.zeros(1,4,dtype=torch.float32)
        
        return pc_pair_th, pose, Tr

    
    def generate_pairs(self, point_clouds, group_matrix):
        '''
        INPUT:
            Point Clouds: <KxNx3>
            group_matrix: <KxN>
            K: Total no. of point clouds in a given sequence
        OUTPUT:
            Point cloud pairs: <K*(G-1)x2xNx3>
        '''

        K, N, c = point_clouds.shape
        G = self.group_matrix.shape[1]

        print(self.group_matrix.shape, point_clouds.shape)

        pcd_pairs = torch.zeros((K*(G-1), 2, N, c))   # <K*(G-1)x2xNx3>
        gt_pairs = torch.zeros((K*(G-1), 4, 4))   # <K*(G-1)x4x4>

        for i in range(1, K+1):
            print(i)
            indices = self.group_matrix[i-1]
            pcd = self.point_clouds[indices, :, :]  # <GxNx3>
            gt = self.gt_pose[indices, :]  # <Gx6>

            centre_frame = pcd[0, :, :].unsqueeze(0)      # <1xNx3>
            centre_frame = centre_frame.repeat(G-1, 1, 1).unsqueeze(1) #<G-1x1xNx3>

            centre_frame_gt = gt[0, :]

            spatial_neigbors = pcd[1:, :, :].unsqueeze(1) # <G-1x1xNx3>

            spatial_neigbors_gt = gt[1:, :]
            spatial_neigbors_Trans = []   

            for j in range(G-1):
                T_rel = self.get_position_transform(pos0=spatial_neigbors_gt[j, :], pos1=centre_frame_gt) #<4x4>
                spatial_neigbors_Trans.append(T_rel)
                
            spatial_neigbors_Trans = np.asarray(spatial_neigbors_Trans) #<(G-1)x4x4>


            pairs = torch.cat((spatial_neigbors, centre_frame), 1) #<G-1x2xNx3>
            pcd_pairs[(i-1)*(G-1):i*(G-1), :, :, :] = pairs 
            gt_pairs[(i-1)*(G-1):i*(G-1), :, :] = torch.from_numpy(spatial_neigbors_Trans).float()
            

        return pcd_pairs, gt_pairs


    def __len__(self):
        return self.n_pc


class KittiEval(Dataset):
    def __init__(self, train_dataset):
        super().__init__()
        self.point_clouds = train_dataset.point_clouds
        self.valid_points = train_dataset.valid_points
        self.init_pose = train_dataset.init_pose
        self.n_pc = train_dataset.n_pc
        self.n_points = train_dataset.n_points
        self.gt_pose = train_dataset.gt_pose

    def __getitem__(self, index):
        pcd = self.point_clouds[index, :, :] # <Nx3>
        init_pose = self.init_pose[index, :] # <4>
        return pcd, init_pose

    def __len__(self):
        return self.n_pc