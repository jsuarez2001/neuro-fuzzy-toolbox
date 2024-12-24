import numpy as np
import torch

from membership_functions import MembershipFunction

class Gaussian_MF(MembershipFunction):
    def __init__(self):
        super(Gaussian_MF, self).__init__()
        self._params = ["mu", "sigma"]
        self._simple_implementation = lambda x, mu, sigma: np.exp(-0.5 * np.power((x - mu)/sigma, 2))

    def forward(self, x, premises):
        return torch.exp(-0.5 * torch.pow((x.unsqueeze(x.dim()) - premises[:, :, 0])/torch.where(premises[:, :, 1] == 0, torch.tensor(1e-6), premises[:, :, 1]), 2))

    def initialize_premises(self, x_train, fuzzy_rules):
        input_size = x_train.shape[1]
        premises = torch.zeros(input_size, fuzzy_rules, len(self._params), dtype=x_train.dtype)
        
        if fuzzy_rules > 1:
            min = torch.min(x_train, dim=0).values
            max = torch.max(x_train, dim=0).values
            stp = (max - min) / (fuzzy_rules - 1)
            for i in range(input_size):
                h = torch.arange(min[i], max[i] + stp[i], stp[i])
                premises[i, :, 0] = h[:fuzzy_rules]
                premises[i, :, 1] = stp[i]/2
        else:
            for i in range(input_size):
                premises[i, :, 0] = torch.mean(x_train[:, i])
                premises[i, :, 1] = torch.std(x_train[:, i])
                
        return premises