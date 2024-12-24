import numpy as np
import torch

from membership_functions import MembershipFunction


class GeneralizedBell_MF(MembershipFunction):
    def __init__(self):
        super(GeneralizedBell_MF, self).__init__()
        self._params = ["a", "b", "c"] # ["width", "slope", "center"]
        self._simple_implementation = lambda x, a, b, c: 1/(1 + np.power(np.abs((x - c)/a), 2*b))

    def forward(self, x, premises):
        return 1/(1 + torch.pow(torch.abs((x.unsqueeze(x.dim()) - premises[:, :, 2])/torch.where(premises[:, :, 0] == 0, torch.tensor(1e-6), premises[:, :, 0])), 2*premises[:, :, 1]))

    def initialize_premises(self, x_train, fuzzy_rules):
        input_size = x_train.shape[1]
        premises = torch.zeros(input_size, fuzzy_rules, len(self._params), dtype=x_train.dtype)
        
        if fuzzy_rules > 1:
            min = torch.min(x_train, dim=0).values
            max = torch.max(x_train, dim=0).values
            stp = (max - min) / (fuzzy_rules - 1)
            for i in range(input_size):
                h = torch.arange(min[i], max[i] + stp[i], stp[i])
                premises[i, :, 2] = h[:fuzzy_rules]
                premises[i, :, 0] = stp[i]/2
                premises[i, :, 1] = 8
        else:
            for i in range(input_size):
                premises[i, :, 2] = torch.mean(x_train[:, i])
                premises[i, :, 0] = torch.std(x_train[:, i])
                premises[i, :, 1] = 8
                
        return premises