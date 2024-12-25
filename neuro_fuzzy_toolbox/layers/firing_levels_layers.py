import torch
import torch.nn as nn


class FiringLevelsLayer(nn.Module):
    def __init__(self, restricted=False):
        super(FiringLevelsLayer, self).__init__()
        self._restricted = restricted

    def forward(self, membership_values):
        if self._restricted:
            w = membership_values.prod(dim=membership_values.dim()-2)
        else:
            w = torch.cat([torch.cartesian_prod(*torch.unbind(t, dim=0)).prod(dim=-1) for t in membership_values]).reshape(-1, membership_values.shape[-1]**membership_values.shape[-2])
        return w



class NormalizationLayer(nn.Module):
    def forward(self, w):
        sum = torch.sum(w, dim=-1, keepdim=True)
        sum[sum == 0] = 1
        w = w/sum
        return w