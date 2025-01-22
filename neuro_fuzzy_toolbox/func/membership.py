import torch
import torch.nn as nn
import numpy as np

from abc import abstractmethod

class MembershipFunction(nn.Module):
    def __init__(self):
        super(MembershipFunction, self).__init__()
        self._params = None
        
        self._max_val_plot = None # For plotting
        self._min_val_plot = None # For plotting
        
    @abstractmethod
    def _simple_numpy_implementation(self, x, *args): # Simple numpy implementation needed for plotting
        pass
        
    @abstractmethod
    def forward(self, x, premises):
        pass
    
    @abstractmethod
    def initialize_premises(self, x_train, fuzzy_rules):
        pass
    
    @abstractmethod
    def random_premises(self, input_size, fuzzy_rules, dtype):
        pass
    
    @abstractmethod
    def _grow_new_premise_parameters(self, means, stds):
        pass
    
    @abstractmethod
    def _split_premise_parameters(self, premises):
        pass


class Gaussian_MF(MembershipFunction):
    def __init__(self):
        super(Gaussian_MF, self).__init__()
        self._params = ["mu", "sigma"]
        
        self._min_val_plot = -2
        self._max_val_plot = 2
        
    def _simple_numpy_implementation(self, x, mu, sigma):
        return np.exp(-0.5 * np.power((x - mu)/sigma, 2))

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
    
    def random_premises(self, input_size, fuzzy_rules, dtype):
        random_premises = 2 * torch.rand(input_size, fuzzy_rules, len(self._params), dtype=dtype) - 1
        random_premises[:, :, 1] = torch.abs(random_premises[:, :, 1])
        return random_premises

    def _grow_new_premise_parameters(self, means, stds):
        """_summary_

        Args:
            means (torch.tensor): 2D tensor with the means of a data group, shape (new fuzzy rules, input size)
            stds (torch.tensor): 2D tensor with the standard deviations of a data group, shape (new fuzzy rules, input size)

        Returns:
            torch.tensor: a 3D tensor with the new premises
        """
        return torch.cat((means.t().unsqueeze(2), stds.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        split1 = torch.clone(premises)
        split1[:,:,0] += premises[:,:,1]/2
        split1[:,:,1] /= 2
        
        split2 = torch.clone(premises)
        split2[:,:,0] -= premises[:,:,1]/2
        split2[:,:,1] /= 2
        
        return torch.cat((split1, split2), dim=1)


class GeneralizedBell_MF(MembershipFunction):
    def __init__(self):
        super(GeneralizedBell_MF, self).__init__()
        self._params = ["a", "b", "c"] # ["width", "slope", "center"]

        self._min_val_plot = -2

        self._max_val_plot = 2
    
    def _simple_numpy_implementation(self, x, a, b, c):
        return 1/(1 + np.power(np.abs((x - c)/a), 2*b))
        
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
                premises[i, :, 1] = 4.
        else:
            for i in range(input_size):
                premises[i, :, 2] = torch.mean(x_train[:, i])
                premises[i, :, 0] = torch.std(x_train[:, i])
                premises[i, :, 1] = 4.
                
        return premises
    
    def random_premises(self, input_size, fuzzy_rules, dtype):
        random_premises = 2 * torch.rand(input_size, fuzzy_rules, len(self._params), dtype=dtype) - 1
        random_premises[:, :, :2] = torch.abs(random_premises[:, :, :2])
        random_premises[:, :, 1] += 1.
        return random_premises
    
    def _grow_new_premise_parameters(self, means, stds):
        return torch.cat((stds.t().unsqueeze(2), (torch.ones_like(stds) * 4.).t().unsqueeze(2), means.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        split1 = torch.clone(premises)
        split1[:,:,0] /= 2
        split1[:,:,2] += split1[:,:,0]
        
        split2 = torch.clone(premises)
        split2[:,:,0] /= 2
        split2[:,:,2] -= split2[:,:,0]
        
        return torch.cat((split1, split2), dim=1)
