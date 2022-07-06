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
from data_loader import kitti_data
from lib.timer import AverageMeter
import logging

# Code runs deterministically 
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='exp2_train_4',help='experiment name')
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
    parser.add_argument('--num_keypoints', type=int, default=1024, metavar='N',
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
    train_dataset = kitti_data.Kitti('D:\kitti_group', opt.traj, opt.voxel_size, init_pose=None, 
            group=True, group_size=9)
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=8)
    #test_loader =  DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
    loss_fn = eval('loss2.'+opt.loss)
    print('creating model......')

    model = deepmapping2.DeepMappingKITTI(loss_fn=loss_fn, args=opt,n_samples=opt.n_samples).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    lr_step = [25, 50, 75]
    scheduler = MultiStepLR(optimizer,
                                milestones=[int(i) for i in lr_step],
                                gamma=0.3)

    if opt.model is not None:
        utils.load_checkpoint(opt.model, model, optimizer)

    print('start training')
    torch.cuda.empty_cache()

    # Start training 
    trans_error_min = np.Inf



    for epoch in range(opt.n_epochs):

        
        training_loss = 0.0
    
        model.train()
        print("Training epoch:", epoch)
        

        for index, (obs_batch, pose_batch) in enumerate(train_loader):

            obs_batch = obs_batch.to(device)
            pose_batch = pose_batch.to(device)

            loss = model(obs_batch, pose_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            #print('training loss', training_loss)
        
        training_loss_epoch = training_loss / len(train_loader)
        print('[{}/{}], training loss: {:.4f}'.format(
                epoch+1,opt.n_epochs,training_loss_epoch))

        scheduler.step()

        if (epoch+1) % opt.log_interval == 0:
            #print('[{}/{}], training loss: {:.4f}'.format(
            #    epoch+1,opt.n_epochs,training_loss_epoch))

            source_pc_est_np = []
            template_pc_np = []
            R_est_np = []
            t_est_np = []

            transformation_gt = []



            with torch.no_grad():
                model.eval()
                for index, (obs_batch_test, pose_batch_test) in enumerate(train_loader):
                    obs_batch_test = obs_batch_test.to(device)
                    pose_batch_test = pose_batch_test.to(device)

                    model(obs_batch_test, pose_batch_test)
                    
                    source_pc_est_np.append(model.source_pc_transform.cpu().detach().numpy())
                    template_pc_np.append(model.tmp.cpu().detach().numpy())
                    R_est_np.append(model.R_est.cpu().detach().numpy())
                    t_est_np.append(model.t_est.unsqueeze(1).cpu().detach().numpy())

                R_est_np = np.concatenate(R_est_np)
                t_est_np = np.concatenate(t_est_np)
                
                save_name = os.path.join(checkpoint_dir,'model_best.pth')
                utils.save_checkpoint(save_name,model,optimizer)

                source_pc_est_np = np.concatenate(source_pc_est_np)
                template_pc_np = np.concatenate(template_pc_np)

                kwargs = {'e':epoch+1}

                utils.plot_global_point_cloud_KITTI(source_pc_est_np[1, :, :], template_pc_np[1, :, :], checkpoint_dir, plot_num=1, **kwargs)
                utils.plot_global_point_cloud_KITTI(source_pc_est_np[100, :, :], template_pc_np[100, :, :], checkpoint_dir, plot_num=2, **kwargs)
                utils.plot_global_point_cloud_KITTI(source_pc_est_np[800, :, :], template_pc_np[800, :, :], checkpoint_dir, plot_num=3, **kwargs)
                 