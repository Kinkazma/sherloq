# Local SHERLOQ adaptation; original: models/external/clone_detectors/03_cmsegnet/predict_folder/Cor.py
# Source SHA256: 6026290db2fc0a0d88eab3bf8d290b3bdc067c2c58ec64f6da9b46830648a049
"""
Created on Tue May 14 22:19:34 2024

@author: user
"""


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


class ZeroWindow:
    def __init__(self):
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1):
        sigma = h * rat_s, w * rat_s
        #print(sigma)
        #sigma = 32,32
        # c = h * w
        b, c, h2, w2 = x_in.shape
        key = (tuple(x_in.shape), rat_s, str(x_in.device), x_in.dtype)
        if key not in self.store:
            ind_r = torch.arange(h2).float()
            ind_c = torch.arange(w2).float()
            ind_r = ind_r.view(1, 1, -1, 1).expand_as(x_in)
            ind_c = ind_c.view(1, 1, 1, -1).expand_as(x_in)

            # center
            c_indices = torch.from_numpy(np.indices((h, w))).float()
            c_ind_r = c_indices[0].reshape(-1)
            c_ind_c = c_indices[1].reshape(-1)

            cent_r = c_ind_r.reshape(1, c, 1, 1).expand_as(x_in)
            cent_c = c_ind_c.reshape(1, c, 1, 1).expand_as(x_in)

            def fn_gauss(x, u, s):
                return torch.exp(-(x - u) ** 2 / (2 * s ** 2))
            #print(sigma)
            gaus_r = fn_gauss(ind_r, cent_r, sigma[0])
            gaus_c = fn_gauss(ind_c, cent_c, sigma[1])
            out_g = 1 - gaus_r * gaus_c
            out_g = out_g.to(x_in.device)
            self.store[key] = out_g
        else:
            out_g = self.store[key]
        out = out_g * x_in
        return out


def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class Corr(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk
        self.zero_window = ZeroWindow()
        self.alpha = nn.Parameter(torch.tensor(5., dtype=torch.float32))

    def forward(self, x):
        b, c, h1, w1 = x.shape
        h2 = h1
        w2 = w1

        xn = F.normalize(x, p=2, dim=-3)
        x_aff_o = torch.matmul(xn.permute(0, 2, 3, 1).view(b, -1, c), xn.view(b, c, -1))

        x_aff = self.zero_window(x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(b, h1 * w1, h2 * w2)
        x_c = F.softmax(x_aff * self.alpha, dim=-1) * F.softmax(x_aff * self.alpha, dim=-2)
        x_c = x_c.reshape(b, h1, w1, h2, w2)

        xc_o = x_c.view(b, h1 * w1, h2, w2)
        val = get_topk(xc_o, k=self.topk, dim=-3)

        return val


