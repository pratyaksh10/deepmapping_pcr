import os
import argparse
import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import mean_squared_error

from models import utils, loss2, deepmapping2
from data_loader import kitti_data, kitti_data_test
from lib.timer import AverageMeter
import logging

import open3d as o3d

# Code runs deterministically 
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)


def metric(R_pred, t_pred, T_gt):

    k = T_gt.shape[0]
    rte_meter, rre_meter = AverageMeter(), AverageMeter()

    
    identity = np.eye(3)
    rre = mean_squared_error(np.transpose(R_pred.reshape(3,3)) @ T_gt[:3, :3].reshape(3,3), identity)
    rte = mean_squared_error(t_pred.reshape(3, 1), T_gt[:3, 3].reshape(3, 1))

    return rre, rte

def icp_o3d(src,dst,voxel_size=1):
    '''
    Don't support init_pose and only supports 3dof now.
    Args:
        src: <Nx3> 3-dim moving points
        dst: <Nx3> 3-dim fixed points
        n_iter: a positive integer to specify the maxium nuber of iterations
        init_pose: [tx,ty,theta] initial transformation
        torlerance: the tolerance of registration error
        metrics: 'point' or 'plane'
        
    Return:
        src: transformed src points
        R: rotation matrix
        t: translation vector
        R*src + t
    '''
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    treg = o3d.t.pipelines.registration
    src_pcd = o3d.t.geometry.PointCloud(device)
    src_pcd.point["positions"] = o3d.core.Tensor(np.asarray(src), dtype, device)
    src_pcd.estimate_normals()
    dst_pcd = o3d.t.geometry.PointCloud(device)
    dst_pcd.point["positions"] = o3d.core.Tensor(np.asarray(dst), dtype, device)
    dst_pcd.estimate_normals()

    voxel_sizes = o3d.utility.DoubleVector([voxel_size])

    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=1e-5,
                                    relative_rmse=1e-5,
                                    max_iteration=30),
        # treg.ICPConvergenceCriteria(1e-5, 1e-5, 30),
        # treg.ICPConvergenceCriteria(1e-6, 1e-6, 50)
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.utility.DoubleVector([3 * voxel_size])

    # Initial alignment or source to target transform.
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64)

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPlane()

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    save_loss_log = True

    registration_ms_icp = treg.multi_scale_icp(src_pcd, dst_pcd, voxel_sizes,
                                           criteria_list,
                                           max_correspondence_distances,
                                           init_source_to_target, estimation,
                                           save_loss_log)

    transformation = registration_ms_icp.transformation
    R = transformation[:3, :3]
    t = transformation[:3, 3:]
    print(transformation.numpy())
    return R.numpy(), t.numpy()

if __name__ == '__main__':    

    pair_pcs = np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\group_pairs.npy')
    pair_gt_trans = np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\group_gt_trans.npy')

    K = pair_pcs.shape[0]

    RRE, RTE = [], []

    for index in range(K):

        pcd_pair = pair_pcs[index,:,:, :]
        T_gt_index = pair_gt_trans[index, :, :]

        src_index = pcd_pair[0, :, :]
        tgt_index = pcd_pair[1, :, :]

        r_pred, t_pred = icp_o3d(src_index, tgt_index)

        rre, rte = metric(r_pred, t_pred, T_gt_index)

        print('---------------------------')
        print('rre:', rre)
        print('rtr:', rte)
        print('---------------------------')

        RRE.append(rre)
        RTE.append(rte)
        print(T_gt_index)
        print()

    average_rre = sum(RRE) / len(RRE)
    average_rte = sum(RTE) / len(RTE)

    print('Average RRE:', average_rre)
    print('Average RTE:', average_rte)

    
   