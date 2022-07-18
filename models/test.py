import torch
import numpy as np

'''
a = torch.randn(4, 5000, 3)
b = torch.randn(4, 5000, 3)
k = a.shape[2]
print(k)
c = a[1, 10, :] - b[1, 10, :] 
c = c.view(k, -1)
print(c.shape)
print(torch.matmul(c.transpose(0,1), c)[0][0])
print(float(torch.matmul(c.transpose(0,1), c)[0][0]))
d = np.float64(torch.matmul(c.transpose(0,1), c)[0][0])
sigma = 100
D = 3
G_sigma = (1/(2*np.pi*sigma)**(1/D))*np.exp((d/(2*sigma**2)))
print(G_sigma)
'''


path = r'D:\kitti_group\2011_09_30_drive_0018_sync_tfvpr\gt_pose.npy'
radius = 6378137

gt_pose = np.load(path)
gt_pose[:, :2] *= np.pi / 180
lat_0 = gt_pose[0, 0]
gt_pose[:, 1] *= radius * np.cos(lat_0)
gt_pose[:, 0] *= radius
gt_pose[:, 1] -= gt_pose[0, 1]
gt_pose[:, 0] -= gt_pose[0, 0]


gt = gt_pose[:, [1, 0, 2, 3, 4, 5]]

print(gt[0:2, :].shape)
def rot3d(axis, angle):
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

def pos_transform(pos):
    x, y, z, rx, ry, rz = pos
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(rot3d(0, rx), rot3d(1, ry)), rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT
'''
T = []
for i in range(gt.shape[0]):

    pos = gt[i, :]
    T.append(pos_transform(pos))

T = np.asarray(T)
'''
save_path = r'D:\kitti_group\2011_09_30_drive_0018_sync_tfvpr\gt_trans.npy'
#np.save(save_path, T)
T = np.load(save_path)

print(T[0, :, :].shape)


