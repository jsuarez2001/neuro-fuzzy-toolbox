import torch
import torch.nn as nn
from torch.nn import Parameter

import pandas as pd

from neuro_fuzzy_toolbox_old.func.membership import gaussian2

class FuzzifyLayer(nn.Module):
    '''
    Fuzzification layer of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Attributes:
    - input_size (int): The size of the input features.
    - dtype (torch.dtype): The data type of the input data (used to initialize the premises with it)
    - mf (function): Membership function to use (default: gaussian2).
    - params (list): List of parameter names for the membership function (default: ['mu', 'sigma']).
    - premises (torch.nn.Parameter): Trainable parameters for the fuzzification layer.

    Methods:
    - __init__: Initializes a new FuzzifyLayer instance.
    - init_premises: Initializes the premises based on input training data.
    - forward: Performs a forward pass through the fuzzification layer.
    - premises_structure: Prints the structure of the fuzzy premises.

    Example Usage:
    >>> input_data = torch.randn((3, 4))
    >>> fuzzify_layer = FuzzifyLayer(input_data, init_rules=3)
    >>> membership_values = fuzzify_layer(input_data)

    '''
    def __init__(self, x_train, init_rules=1, mf=gaussian2, params=['mu', 'sigma'], init_mode=0):
        """
        Initializes a new FuzzifyLayer instance.

        Parameters:
        - x_train (torch.tensor): Input training data.
        - init_rules (int): The number of initial fuzzy rules (default: 1).
        - mf (function): Membership function to use (default: gaussian2).
        - params (list): List of parameter names for the membership function (default: ['mu', 'sigma']).
        - init_mode (int): Numeric flag for initializing the fuzzy premises (default: 0, meaning it will be initialized based on the input data,
                           otherwise it will be initialized randomly).

        """
        super(FuzzifyLayer, self).__init__()
        self.input_size = x_train.shape[1]
        self.dtype = x_train.dtype
        self.mf = mf
        self.params = params

        # Initialize premises
        if init_mode != 0: #random
            prems = 2 * torch.rand(self.input_size, init_rules, len(params), dtype=self.dtype) - 1
            prems[:,:,1] = (prems[:,:,1] + 1)/2
            self.premises = Parameter(prems, requires_grad=True)
        else: #based on the training input data
            self.premises = Parameter(self.init_premises(x_train, init_rules), requires_grad=True)


    def init_premises(self, x_train, init_rules):
        """
        Initializes the fuzzy premises based on input training data.

        Parameters:
        - x_train (torch.Tensor): Training data for initializing fuzzy premises.
        - init_rules (int): The number of initial fuzzy rules.

        Returns:
        - torch.Tensor: Initialized fuzzy premises.

        """
        premises = torch.zeros(self.input_size, init_rules, len(self.params), dtype=x_train.dtype)
        if init_rules > 1:
            min = torch.min(x_train, dim=0).values
            max = torch.max(x_train, dim=0).values
            stp = (max - min) / (init_rules - 1)
            for i in range(self.input_size):
                h = torch.arange(min[i], max[i] + stp[i], stp[i])
                premises[i, :, 0] = h[:init_rules]
                premises[i, :, 1] = stp[i]/2
        else:
            for i in range(self.input_size):
                premises[i, :, 0] = torch.mean(x_train[:, i])
                premises[i, :, 1] = torch.std(x_train[:, i])
        return premises


    def forward(self, x):
        """
        Performs a forward pass through the fuzzification layer.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor (membership values).

        """
        return self.mf(x.unsqueeze(x.dim()), self.premises)


    @property
    def premises_structure(self):
        """
        return the structure of the fuzzy premises on a dataframe.

        """
        data = {}
        df = pd.DataFrame()
        rules = ['rule {}'.format(i) for i in range(1, self.premises.data.shape[1]+1)]

        for i in range(self.input_size):
            for j in range(len(self.params)):
                column = pd.Series(self.premises.data[i,:,j], index=rules, name=self.params[j] + f' (x{i})', )
                df[self.params[j] + f' (x{i})'] = column

        return df