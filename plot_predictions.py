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
from matplotlib import pyplot as plt

from models import utils, loss2, deepmapping2
from data_loader import kitti_data
from lib.timer import AverageMeter
import logging
import open3d as o3d
from scipy.spatial.transform import Rotation

# Code runs deterministically 
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

def generate_rand_trans(x_lim=10.0, y_lim=1.0, z_lim=0.1):
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        trans [3]
    '''
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)

    rand_trans = np.array([rand_x, rand_y, rand_z])

    return rand_trans

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def generate_rand_rotm(x_lim=5.0, y_lim=5.0, z_lim=180.0):
 
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)

    rand_eul = np.array([rand_z, rand_y, rand_x])
    r = Rotation.from_euler('zyx', rand_eul, degrees=True)
    rotm = r.as_matrix()
    return rotm

def unsupervised_pairwise_reg(pairwise_batch, pose_batch, Tr, model, device, model_path=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']) 
    pcd0 = pairwise_batch[0, :, :]
    pcd1 = pairwise_batch[1, :, :]
    
    
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
    pair_pc = torch.from_numpy(pc_pair).float()
    pair_pc = pair_pc.unsqueeze(0).to(device)

    pose_batch = torch.Tensor(pose_batch)
    pose_batch = pose_batch.unsqueeze(0).to(device)
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        model(pair_pc, pose_batch)

        rotation_batch = model.R_est
        translation_batch = model.t_est

        #src_transformed = model.source_pc_transform
        #tmp = model.tmp

    return pair_pc, rotation_batch, translation_batch#, src_transformed, tmp


def plot_global_point_cloud_KITTI(source_pc, template_pc, save_dir, plot='icp'):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    if torch.is_tensor(source_pc):
        source_pc = source_pc.cpu().detach().numpy()
    if torch.is_tensor(template_pc):
        template_pc = template_pc.cpu().detach().numpy()

    file_name = 'global_map_pose_' + plot

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

def plot_gt_transform(pair_pc, R_gt, t_gt, plot='icp'):
    src = torch.Tensor(pair_pc[0, :, :]).unsqueeze(0)
    tgt = torch.Tensor(pair_pc[1, :, :]).unsqueeze(0)
    R_gt = torch.Tensor(R_gt).unsqueeze(0)
    t_gt = torch.Tensor(t_gt).unsqueeze(0)

    print(src.shape, tgt.shape, R_gt.shape, t_gt.shape)
    src_transformed = utils.transform_to_global_KITTI(R_gt.transpose(2, 1).contiguous(), t_gt, src)

    save_path = r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\results\gt\icp'
    plot_global_point_cloud_KITTI(src_transformed.squeeze(0), tgt.squeeze(0), save_path, plot=plot)

    return None

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
    rand_rotm = generate_rand_rotm(1.0, 1.0, 180.0)
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
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='exp2_vis',help='experiment name')
    parser.add_argument('-e','--n_epochs',type=int,default=120,help='number of epochs')
    parser.add_argument('-b','--batch_size',type=int,default=2,help='batch_size')
    parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')
    parser.add_argument('-n','--n_samples',type=int,default=45,help='number of sampled unoccupied points along rays')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('-d','--root',type=str,default='/mnt/NAS/home/xinhao/deepmapping/main/data/kitti/',help='root path')
    parser.add_argument('-t','--traj',type=str,default='2011_09_30_drive_0018_sync_tfvpr',help='trajectory file folder')
    parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
    parser.add_argument('-m','--model', type=str, default=None,help='pretrained model name')
    parser.add_argument('-i','--init', type=str, default=None,help='init pose')
    parser.add_argument('--log_interval',type=int,default=1,help='logging interval of saving results')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=256, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_keypoints', type=int, default=700, metavar='N',
                        help='Number of key poinits')
    parser.add_argument('--des_dim', type=int, default=256, metavar='N',
                        help='Neiborhood descriptor dimension')
    parser.add_argument('--k', type=int, default=4, metavar='N',
                        help='No. of nearest neighbors')
    parser.add_argument('--dim', type=int, default=16, metavar='N',
                        help='Dim')
    opt = parser.parse_args()
    checkpoint_dir = os.path.join('./results/', opt.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save parser arguments
    utils.save_opt(checkpoint_dir, opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print('loading dataset........')

    #test_dataset = kitti_data.Kitti('D:\kitti_group', opt.traj, opt.voxel_size, init_pose=None,
    #        group=True, group_size=9)
    #test_loader = DataLoader(test_dataset, batch_size=8, num_workers=8)

    #test_loader =  DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
    loss_fn = eval('loss2.'+opt.loss)
    print('creating model......')

    model = deepmapping2.DeepMappingKITTI(loss_fn=loss_fn, args=opt,n_samples=opt.n_samples)
    PATH = r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\results\gt\our_model\model_best.pth'

    #PATH = r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\results\exp2_train_4\model_best.pth'

    pair_gt_trans= np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\group_gt_trans_icp.npy')
    pair_pcs  = np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\pc_pairs_icp.npy')


    pc_pair_1 = pair_pcs[7, :, :, :] #<2, N, 3>
    pc_pair_2 = pair_pcs[9000, :, :, :]
    gt_1 = pair_gt_trans[7, :, :]    #<4,4>
    gt_2 = pair_gt_trans[9000, :, :]
    
    print(pc_pair_2[0, :, :])
    print(pc_pair_2[1, :, :])

    #r_icp, t_icp = icp_o3d(pc_pair_1)
    r_icp2, t_icp2 = icp_o3d(pc_pair_2)
    #print(r_icp.shape, gt_1[:3, :3].shape, t_icp.reshape(3,).shape, gt_1[:3, 3].shape)
    #plot_gt_transform(pc_pair_1, r_icp, t_icp.reshape(3,), plot='gt4')
    plot_gt_transform(pc_pair_2, r_icp2, t_icp2.reshape(3,), plot='icp_rot2')
    #plot_gt_transform(pc_pair_1, gt_1[:3, :3], gt_1[:3, 3], 3)  



    #pc_pair_1 = torch.Tensor(pc_pair_1) #<2, N, 3>
    pose_batch_test = np.zeros((1,4))
    #gt_1 = torch.Tensor(gt_1) #<4,4>

    #pc_pair_1, r_pred_batch, t_pred_batch = unsupervised_pairwise_reg(pc_pair_2, pose_batch_test, gt_2, model, device, model_path=PATH)

    #plot_gt_transform(pc_pair_1.squeeze(0).to('cpu'), r_pred_batch.squeeze(0).to('cpu'), t_pred_batch.squeeze(0).to('cpu'), 'our_model_rot2')









