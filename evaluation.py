import os
import argparse
import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader

from models import utils, loss, deepmapping2
from data_loader_kitti import kitti_loader_exp2
from lib.timer import AverageMeter

# Code runs deterministically 
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

    

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='evaluation',help='experiment name')
parser.add_argument('-e','--n_epochs',type=int,default=100,help='number of epochs')
parser.add_argument('-b','--batch_size',type=int,default=1,help='batch_size')
parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')
parser.add_argument('-n','--n_samples',type=int,default=30,help='number of sampled unoccupied points along rays')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-m','--model', type=str, default='./results/exp1_train/model_best.pth',help='pretrained model name')
parser.add_argument('-i','--init', type=str, default=None,help='init pose')
parser.add_argument('--log_interval',type=int,default=1,help='logging interval of saving results')
parser.add_argument('--cosm',type=bool,default=True,help='correntropy similarity matrix')
parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Embedding nn to use, [pointnet, dgcnn]')
parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                    choices=['identity', 'transformer'],
                    help='Attention-based pointer generator to use, [identity, transformer]')
parser.add_argument('--head', type=str, default='svd', metavar='N',
                    choices=['mlp', 'svd', ],
                    help='Head to use, [mlp, svd]')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
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

opt = parser.parse_args()
checkpoint_dir = os.path.join('./results/', opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Save parser arguments
utils.save_opt(checkpoint_dir, opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print('loading dataset........')
dataset = kitti_loader_exp2.KITTINMPairDataset(phase='train')
#print('loading completed!')
print("Number of point cloud pairs for training:", len(dataset.files))
loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
#test_loader =  DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
loss_fn = eval('loss.'+opt.loss)
print('loading model......')

model = deepmapping2.DeepMappingKITTI(loss_fn=loss_fn, args=opt,n_samples=opt.n_samples).to(device)
optimizer = optim.SGD(model.parameters(), lr=opt.lr)

utils.load_checkpoint(opt.model, model, optimizer)

source_pc_est_np = []
template_pc_np = []
R_est_np = []
t_est_np = []

count = 0 


success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()

with torch.no_grad():
    model.eval()
    for index, (obs_batch_test, pose_batch_test) in enumerate(loader):
        obs_batch_test = obs_batch_test.to(device)
        pose_batch_test = pose_batch_test.to(device)
        model(obs_batch_test,pose_batch_test)

        source_pc_est_np.append(model.source_pc_transform.cpu().detach().numpy())
        template_pc_np.append(model.tmp.cpu().detach().numpy())
        R_est_np.append(model.R_est.cpu().detach().numpy())
        t_est_np.append(model.t_est.cpu().detach().numpy())

        print('count:', count)

        test = 1 
        kwargs = {'e':test}
        count += 1

    
    source_pc_est_np = np.concatenate(source_pc_est_np)
    template_pc_np = np.concatenate(template_pc_np)
    utils.plot_global_point_cloud_KITTI(source_pc_est_np[0, :, :], template_pc_np[0, :, :], checkpoint_dir, **kwargs)




