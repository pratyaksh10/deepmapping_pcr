import torch
import numpy as np

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