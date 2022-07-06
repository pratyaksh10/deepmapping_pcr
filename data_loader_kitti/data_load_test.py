import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import open3d as o3D
import glob
from scipy.spatial.transform import Rotation

def apply_rotation(pc, R):
    pc = pc @ R.T
    return pc

def generate_rand_rotm(x_lim=0, y_lim=0.1745, z_lim=0):
    
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        rotm: [3,3]
    '''
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)

    rand_eul = np.array([rand_z, rand_y, rand_x])
    r = Rotation.from_euler('zyx', rand_eul, degrees=False)
    rotm = r.as_matrix()
    return rotm

class KITTI(Dataset):
    def __init__(self):
        self.point_clouds_dir = r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping-master\data_loader_kitti\kitti_point_clouds_voxel_0.5.npy'
        
        point_clouds = np.load(self.point_clouds_dir, allow_pickle=True)
        point_clouds = point_clouds[:10, ::3, :]
        st_point_clouds = [] # Source-Template pairs
        for i in range(point_clouds.shape[0] - 6):

            source_pc = point_clouds[i+5]
            template_pc = point_clouds[i]
            
            # Augmentation with random rotation
            st_point_clouds.append([source_pc, template_pc])

        st_point_clouds = np.asarray(st_point_clouds)
        # number of point clouds in each point cloud
        self.n_points = st_point_clouds.shape[2]

        print(st_point_clouds.shape)
        self.point_clouds = torch.from_numpy(st_point_clouds).float() #<Npcx2xNx3>

        
        # shape of the dataset 
        self.dataset_shape = self.point_clouds.shape


    def __getitem__(self, index):
        pcd = self.point_clouds[index,:,:, :] #<2xNx3>, <source_pc, template_pc>
        pose = torch.zeros(1,4,dtype=torch.float32)
        return pcd, pose
    
    def __len__(self):
        return len(self.point_clouds)