from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from .networks import LNet2, MLP
from .utils import transform_to_global_KITTI

def get_M_net_inputs_labels(occupied_points, unoccupited_points):
    """
    get global coord (occupied and unoccupied) and corresponding labels
    """
    n_pos = occupied_points.shape[1]
    inputs = torch.cat((occupied_points, unoccupited_points), 1)
    bs, N, _ = inputs.shape

    gt = torch.zeros([bs, N, 1], device=occupied_points.device)
    gt.requires_grad_(False)
    gt[:, :n_pos, :] = 1
    return inputs, gt


def sample_unoccupied_point(local_point_cloud, n_samples, center):
    """
    sample unoccupied points along rays in local point cloud
    local_point_cloud: <BxLxk>
    n_samples: number of samples on each ray
    center: location of sensor <Bx1xk>
    """
    bs, L, k = local_point_cloud.shape
    #print("shape before:", center.shape)
    center = center.expand(-1,L,-1) # <BxLxk>
    #print('shape centre', center.shape)
    unoccupied = torch.zeros(bs, L * n_samples, k,
                             device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        unoccupied[:, (idx - 1) * L:idx * L, :] = center + (local_point_cloud-center) * fac
    return unoccupied

class DeepMappingKITTI(nn.Module):
    def __init__(self, loss_fn, args, n_obs=500, n_samples=45, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMappingKITTI, self).__init__()
        self.n_obs = n_obs
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LNet2(args)
        self.occup_net = MLP(dim)

    def forward(self, obs_local, sensor_pose):
        # obs_local: <Bx2xNx3>
        # sensor_pose: init pose <Bx1x3>
        self.obs_local = deepcopy(obs_local)
        src = obs_local[:, 0, :, :]    #<bx1xNx3>
        self.src = src.squeeze(1)      #<bxNx3>

        tmp = obs_local[:, 1, :, :]    #<bx1xNx3>
        self.tmp = tmp.squeeze(1)      #<bxNx3>

        self.R_est, self.t_est, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn = self.loc_net(self.obs_local)

        self.source_pc_transform = transform_to_global_KITTI(self.R_est, self.t_est, self.src) #<bxNx3>
        self
        
        if self.training:

            sensor_center = sensor_pose[:,:,:3]
            self.source_keypoints_transform = transform_to_global_KITTI(self.R_est, self.t_est, src_keypoints) #<bxkey_pointsx3>
            self.unoccupied_local = sample_unoccupied_point(
                tgt_keypoints, self.n_samples,sensor_center)
            #self.unoccupied_global = transform_to_global_KITTI(self.R_est, self.t_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.source_keypoints_transform, self.unoccupied_local)
        
            self.occp_prob = self.occup_net(inputs)
            
            source_keypoints_transform = transform_to_global_KITTI(self.R_est, self.t_est, src_keypoints) #<bxkey_pointsx3>

            loss = self.compute_loss(self.source_pc_transform, self.tmp, source_keypoints_transform, tgt_keypoints,
                                        src_keypoints_knn, tgt_keypoints_knn, self.occp_prob, self.gt)
            return loss

    def compute_loss(self, source_pc_transform, tmp, source_keypoints_transform, tgt_keypoints, 
                            src_keypoints_knn, tgt_keypoints_knn, occp_prob, gt):

        if self.loss_fn.__name__ == 'bce_ch':
            loss = self.loss_fn(source_pc_transform, tmp, source_keypoints_transform, tgt_keypoints, 
                                    src_keypoints_knn, tgt_keypoints_knn, occp_prob, gt)  
        return loss