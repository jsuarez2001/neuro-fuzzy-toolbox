import torch
import torch.nn as nn
from torch.nn import Parameter

import pandas as pd

from neuro_fuzzy_toolbox_old.func import weighted_linear

class ConsequentLayer(nn.Module):
    """
    Class for representing the fourth layer (consequent layer) of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Attributes:
    - function (function): Consequent function to use (default: weighted_linear).
    - consequents (torch.nn.Parameter): Trainable parameters for the consequent layer.


    Methods:
    - __init__: Initializes a new ConsequentLayer instance.
    - forward: Performs a forward pass to calculate the consequent layer output.
    - consequents_structure: Prints the structure of the consequent parameters.

    Example Usage:
    >>> input_data = torch.randn((5, 3))  # Assuming input tensor shape (batch_size, num_input_features)
    >>> consequent_layer = ConsequentLayer(input_data.shape[1], input_data.dtype, init_rules=2)
    >>> output = consequent_layer(input_data, weights) # Assuming weight is the tensor obtained from the normalization layer with shape (batch_size, num_rules)

    """
    def __init__(self, input_size, dtype, init_rules=1, function=weighted_linear, outputs=1):
        """
        Initializes a new ConsequentLayer instance.

        Parameters:
        - input_size (int): Number of input features.
        - dtype (torch.dtype): Data type for the consequents.
        - init_rules (int): Number of initial rules.
        - function (callable): Consequent function to apply.

        """
        super(ConsequentLayer, self).__init__()
        self.function = function
        self.consequents = Parameter(2 * torch.rand(outputs, init_rules, input_size + 1, dtype=dtype) - 1, requires_grad=True)


    def forward(self, x, w):
        """
        Performs a forward pass to calculate the consequent layer output.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - w (torch.Tensor): Weights tensor.

        Returns:
        - torch.Tensor: Output tensor (Outputs by rule of the ANFIS model).

        """
        outputs = self.function(x, self.consequents, w)
        return outputs


    @property
    def consequents_structure(self):
        """
        Returns the structure of the consequent parameters on a list of Pandas dataframes.

        """
        dfs = [pd.DataFrame() for _ in range(self.consequents.data.shape[0])]

        rules = ['rule {}'.format(i) for i in range(1, self.consequents.data.shape[1]+1)]

        for o in range(self.consequents.data.shape[0]):
            for i in range(self.consequents.data.shape[2]):
                name=f'c{i} (x{i})'
                if (i == self.consequents.data.shape[2]-1): name=f'c{i}'
                column = pd.Series(self.consequents.data[o,:,i], index=rules, name=name)
                dfs[o][name] = column

        return dfs