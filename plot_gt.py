import os
import argparse
import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from matplotlib import pyplot as plt

from models import utils, loss2, deepmapping2
from data_loader import kitti_data
from lib.timer import AverageMeter
import logging
import open3d as o3d
from scipy.spatial.transform import Rotation


def plot_global_point_cloud_KITTI(source_pc, template_pc, save_dir, plot='gt'):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    if torch.is_tensor(source_pc):
        source_pc = source_pc.cpu().detach().numpy()
    if torch.is_tensor(template_pc):
        template_pc = template_pc.cpu().detach().numpy()

    file_name = 'global_map_pose' + plot

    save_name = os.path.join(save_dir, file_name)

    st_pair = [source_pc, template_pc]

    n = 2
    for i in range(n):

        current_pc = st_pair[i]

        xs = current_pc[:, 0]#[::50]   
        ys = current_pc[:, 1]#[::50]
        zs = current_pc[:, 2]#[::50]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.scatter(xs, ys, zs, s=0.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(elev=40, azim=180)
        #ax.scatter(current_pc[:, 0], current_pc[:, 1], cmap='gray')
        #ax.plot3D(current_pc[:, 0], current_pc[:, 1], current_pc[:, 2], '.')


    ax_2 = plt.gca()
    ax_2.set_ylim(ax_2.get_ylim()[::-1])
    print(save_name)
    #plt.plot(pose[:, 0], pose[:, 1], color='black')
    plt.savefig(save_name)
    plt.close()

def icp_o3d(pc_pair,voxel_size=1):
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

    src = pc_pair[0, :, :]
    dst = pc_pair[1, :, :]

    aug_T = np.zeros((4,4), dtype=np.float32)
    aug_T[3,3] = 1.0
    rand_rotm = generate_rand_rotm(0, 0, 45.0)
    aug_T[:3,:3] = rand_rotm

    src = apply_transform(src, aug_T)


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
                                    max_iteration=100),
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
    print(transformation.numpy(), 'prediction')
    return R.numpy(), t.numpy()

def plot_gt_transform(pair_pc, R_gt, t_gt, plot='icp'):
    src = torch.Tensor(pair_pc[0, :, :]).unsqueeze(0)
    tgt = torch.Tensor(pair_pc[1, :, :]).unsqueeze(0)
    R_gt = torch.Tensor(R_gt).unsqueeze(0)
    t_gt = torch.Tensor(t_gt).unsqueeze(0)

    print(src.shape, tgt.shape, R_gt.shape, t_gt.shape)
    src_transformed = utils.transform_to_global_KITTI(R_gt.transpose(2, 1).contiguous(), t_gt, src)

    save_path = r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\results\gt'
    plot_global_point_cloud_KITTI(src_transformed.squeeze(0), tgt.squeeze(0), save_path, plot=plot)

    return None

def generate_rand_rotm(x_lim=5.0, y_lim=5.0, z_lim=180.0):
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
    r = Rotation.from_euler('zyx', rand_eul, degrees=True)
    rotm = r.as_matrix()
    return rotm

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

if __name__ == '__main__':

    pair_gt_trans = np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\group_gt_trans_01.npy')
    pair_pcs = np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\pc_pairs_01.npy')

    pc_pair_1 = pair_pcs[7, :, :, :] #<2, N, 3>
    pc_pair_2 = pair_pcs[1000, :, :, :]
    gt_1 = pair_gt_trans[7, :, :]    #<4,4>
    gt_2 = pair_gt_trans[1000, :, :]

    

    r_icp, t_icp = icp_o3d(pc_pair_1)
    r_icp2, t_icp2 = icp_o3d(pc_pair_2)

    plot_gt_transform(pc_pair_1, r_icp, t_icp.reshape(3,), 'icp1')
    plot_gt_transform(pc_pair_2, r_icp2, t_icp2.reshape(3,), 'icp2')
    #plot_gt_transform(pc_pair_1, gt_1[:3, :3], gt_1[:3, 3].reshape(3,), 'gt1')  

    #plot_gt_transform(pc_pair_2, gt_2[:3, :3], gt_2[:3, 3].reshape(3,), 'gt2')