import numpy as np

# t1 = np.arange(10) + 1
# m = 10
# t2 = np.arange(4) + 1
# n = 4
# t1 = np.concatenate((t1[:, np.newaxis, ], t1[:, np.newaxis]), axis=1)
# t2 = np.concatenate((t2[:, np.newaxis, ], t2[:, np.newaxis]), axis=1)
# # Lxembed_dim
#
#
# print(t2.shape)
#
# pad_t1 = np.zeros((12, 2))
# pad_t2 = np.zeros((12, 2))
# pad_t1[:10] = t1
# pad_t2[:4] = t2
# # padxembed_dim
# print(pad_t1.shape, pad_t2.shape)h
# T = pad_t1 @ pad_t2.T
# T = T[:m, :n]
# print(T)
import torch

a = torch.rand((64, 2, 10, 32))
a = a.view(64, 2, -1)
print(a.shape)
a = torch.cat(torch.unbind(a, dim=1), dim=0)
print(a.shape)
c = torch.matmul(a, a.T)
print(c.shape)
