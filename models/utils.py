import torch
import numpy as np
import open3d
import os
import json
from scipy.spatial.transform import Rotation

from matplotlib import pyplot as plt

def generate_rand_rotm(x_lim=6.28, y_lim=6.28, z_lim=6.28):
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

def apply_rotation(pc, R):
    pc = pc @ R.T 
    return pc

def save_opt(working_dir, opt):
    """
    Save option as a json file
    """
    opt = vars(opt)
    save_name = os.path.join(working_dir, 'opt.json')
    with open(save_name, 'wt') as f:
        json.dump(opt, f, indent=4, sort_keys=True)


def save_checkpoint(save_name, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, save_name)
    print('model saved to {}'.format(save_name))



def load_checkpoint(save_name, model, optimizer):
    state = torch.load(save_name)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from {}'.format(save_name))


def load_opt_from_json(file_name):
    if os.path.isfile(file_name):
        with open(file_name,'rb') as f:
            opt_dict = json.load(f)
            return opt_dict
    else:
        raise FileNotFoundError("Can't find file: {}. Run training script first".format(file_name))

def plot_global_point_cloud_KITTI(source_pc, template_pc, save_dir, plot_num, **kwargs):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    if torch.is_tensor(source_pc):
        source_pc = source_pc.cpu().detach().numpy()
    if torch.is_tensor(template_pc):
        template_pc = template_pc.cpu().detach().numpy()

    file_name = 'global_map_pose' + str(plot_num)
    if kwargs is not None:
        for k, v in kwargs.items():
            file_name = file_name + '_' + str(k) + '_' + str(v)
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
    #plt.plot(pose[:, 0], pose[:, 1], color='black')
    plt.savefig(save_name)
    plt.close()

def plot_pose_KITTI(pose, save_dir, **kwargs):
    fig = plt.figure(figsize=(7, 6))
    #ax = fig.add_subplot(111, projection='3d')

    if torch.is_tensor(pose):
        pose = pose.cpu().detach().numpy()

    file_name = 'global_pose'
    if kwargs is not None:
        for k, v in kwargs.items():
            file_name = file_name + '_' + str(k) + '_' + str(v)
    save_name = os.path.join(save_dir, file_name)

   
    plt.plot(pose[:, 0], pose[:, 1], pose[:, 2], color='red')
    plt.xlabel('z')
    plt.ylabel('x')
    #ax.set_zlabel('z')
    #ax.view_init(elev=-40, azim=270)

    plt.savefig(save_name)
    plt.close()

def generate_global_trajectory(local_R, local_t):
    # local_R: <k, 3, 3>
    # local_t: <k, 3>
    # global_t: <k, 3>
    global_t = []

    k = local_R.shape[0]

    for i in range(k):
        
        R_temp = np.array([[1, 0, 0], 
                            [0, 1, 0], 
                            [0, 0, 1]])

        for j in range(i):
            
            R_temp = R_temp@local_R[j, :, :]
        global_t.append(R_temp@local_t[i, :])
    
    return np.asarray(global_t)



  












