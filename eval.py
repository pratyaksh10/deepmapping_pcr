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

def transform_to_global_KITTI(R, t, local_pc):
    """
    transform source local coordinate to template coordinate
    Input:
        R: <bx3x3>
        t: <bx3>
        local_pc (source): <BxNx3>

    Output:
        source point cloud after transformation
    """
    N = local_pc.shape[1]

    
    t = t.unsqueeze(1).expand(-1, N, -1) #<bxNx3>
    source_pc_trans = torch.bmm(local_pc, R) + t   #<bxNx3>

    return source_pc_trans

def unsupervised_pairwise_reg(pairwise_batch, pose_batch, model, model_path=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        model.eval()
        model(pairwise_batch, pose_batch)

        rotation_batch = model.R_est
        translation_batch = model.t_est

        src_transformed = model.source_pc_transform
        tmp = model.tmp

    return rotation_batch, translation_batch, src_transformed, tmp

def groud_truth_reg(pairwise_batch, gt_trans_batch):
    B = pairwise_batch.shape[0]
    src = pairwise_batch[:, 0, :, :]    #<bx1xNx3>
    src = src.squeeze(1)      #<bxNx3>

    tmp = pairwise_batch[:, 1, :, :]    #<bx1xNx3>
    tmp = tmp.squeeze(1)      #<bxNx3>

    R_batch = gt_trans_batch[:, :3, :3] #<B, 3, 3>
    t_batch = gt_trans_batch[:, :3, 3].view(B, 3)  #<B, 3>

    src_trans = transform_to_global_KITTI(R_batch, t_batch, src)

    return src_trans, tmp


def metric(R_pred, t_pred, T_gt):

    k = T_gt.shape[0]
    rte_meter, rre_meter = AverageMeter(), AverageMeter()

    for i in range(k):
        identity = np.eye(3)
        rre = mean_squared_error(np.transpose(R_pred[i, :, :].reshape(3,3)) @ T_gt[i, :3, :3].reshape(3,3), identity)
        rte = mean_squared_error(t_pred[i, :].reshape(3, 1), T_gt[i, :3, 3].reshape(3, 1))

        rte_meter.update(rte)
        rre_meter.update(rre)

    print('RTE:', rte_meter.avg)
    print('RRE:', rre_meter.avg)

    return rre_meter.avg, rte_meter.avg


if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='exp2_test_gt',help='experiment name')
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
    parser.add_argument('--num_keypoints', type=int, default=800, metavar='N',
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

    test_dataset = kitti_data.Kitti('D:\kitti_group', opt.traj, opt.voxel_size, init_pose=None, 
            group=True, group_size=9)
    test_loader = DataLoader(test_dataset, batch_size=2, num_workers=4)

    #test_loader =  DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
    loss_fn = eval('loss2.'+opt.loss)
    print('creating model......')

    model = deepmapping2.DeepMappingKITTI(loss_fn=loss_fn, args=opt,n_samples=opt.n_samples).to(device)

    PATH = r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\results\exp2_train_4\model_best.pth'

    RRE, RTE = [], []

    for index, (obs_batch, pose_batch, gt_trans_batch) in enumerate(test_loader):
        obs_batch = obs_batch.to(device)
        pose_batch = pose_batch.to(device)
        gt_trans_batch = gt_trans_batch.to(device)

        r_pred, t_pred, _, _ = unsupervised_pairwise_reg(obs_batch, pose_batch, model, PATH)
        r_pred = r_pred.cpu().detach().numpy()
        t_pred = t_pred.cpu().detach().numpy()

        T_gt = gt_trans_batch.cpu().detach().numpy()
     

        rre_batch, rte_batch = metric(r_pred, t_pred, T_gt)

        print('---------------------------')
        print('rre batch:', rre_batch)
        print('rtr batch:', rte_batch)
        print('---------------------------')

        RRE.append(rre_batch)
        RTE.append(rte_batch)
        

    average_rre = sum(RRE) / len(RRE)
    average_rte = sum(RTE) / len(RTE)

    print('Average RRE:', average_rre)

    print('Maxmimum RRE', max(average_rre))
    
        
    #pair_pcs = np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\group_pairs.npy')
    #pair_gt_trans = np.load(r'C:\Users\praop\OneDrive\Desktop\NYU\AI4CE\code\DeepMapping_pcr\data_loader\group_gt_trans.npy')

    #pair_pcs_test = pair_pcs[0:10, :, :, :]
    #pair_gt_trans = pair_gt_trans[0:10, :, :]

    #pair_pcs_test = torch.from_numpy(pair_pcs_test).float().to(device)
    #pair_gt_trans = torch.from_numpy(pair_gt_trans).float().to(device)
    #pose = torch.zeros(8, 1,4,dtype=torch.float32).to(device)



    '''
    r_pred, t_pred, src_trans, tmp = unsupervised_pairwise_reg(pair_pcs_test, pose, model, PATH)

    r_pred = r_pred.cpu().detach().numpy()
    t_pred = t_pred.cpu().detach().numpy()
    #source_pc_est_np = src_trans.cpu().detach().numpy()
    template_pc_np = tmp.cpu().detach().numpy()
    
    

    src_trans_gt, tmp = groud_truth_reg(pair_pcs_test, pair_gt_trans)
    source_pc_est_np = src_trans_gt.cpu().detach().numpy()
    template_pc_np = tmp.cpu().detach().numpy()

    metric(r_pred, t_pred, pair_gt_trans.cpu().detach().numpy())
    '''

    #epoch = 1
    #kwargs = {'e':epoch+1}

    #utils.plot_global_point_cloud_KITTI(source_pc_est_np[5, :, :], template_pc_np[5, :, :], checkpoint_dir, plot_num=4, **kwargs)
    #utils.plot_global_point_cloud_KITTI(source_pc_est_np[6, :, :], template_pc_np[6, :, :], checkpoint_dir, plot_num=5, **kwargs)
    #utils.plot_global_point_cloud_KITTI(source_pc_est_np[7, :, :], template_pc_np[7, :, :], checkpoint_dir, plot_num=6, **kwargs)



    
    #checkpoint = torch.load(PATH)

    #model.load_state_dict(checkpoint['state_dict'])
   

    #source_pc_est_np = []
    #template_pc_np = []
    #R_est_np = []
    #t_est_np = []

    #transformation_gt = []



    #with torch.no_grad():
    #    model.eval()
    #    for index, (obs_batch_test, pose_batch_test) in enumerate(test_loader):
    #        obs_batch_test = obs_batch_test.to(device)
    #        pose_batch_test = pose_batch_test.to(device)

    #        model(obs_batch_test, pose_batch_test)
            
    #        source_pc_est_np.append(model.source_pc_transform.cpu().detach().numpy())
    #        template_pc_np.append(model.tmp.cpu().detach().numpy())
    #        R_est_np.append(model.R_est.cpu().detach().numpy())
    #        t_est_np.append(model.t_est.unsqueeze(1).cpu().detach().numpy())

    #    R_est_np = np.concatenate(R_est_np)
    #    t_est_np = np.concatenate(t_est_np)
        
  

    #    source_pc_est_np = np.concatenate(source_pc_est_np)
    #    template_pc_np = np.concatenate(template_pc_np)

    #    epoch = 1
    #    kwargs = {'e':epoch+1}

    #    utils.plot_global_point_cloud_KITTI(source_pc_est_np[1, :, :], template_pc_np[1, :, :], checkpoint_dir, plot_num=1, **kwargs)
    #    utils.plot_global_point_cloud_KITTI(source_pc_est_np[10, :, :], template_pc_np[10, :, :], checkpoint_dir, plot_num=2, **kwargs)
    