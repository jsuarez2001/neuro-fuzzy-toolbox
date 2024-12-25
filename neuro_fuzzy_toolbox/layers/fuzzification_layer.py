import torch
import torch.nn as nn
from torch.nn import Parameter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF

class FuzzificationLayer(nn.Module):
    
    def __init__(self, x_train, init_fuzzy_rules=1, membership_function=GeneralizedBell_MF, init_mode=0):
        super(FuzzificationLayer, self).__init__()
        # Input data info
        self._input_size = x_train.shape[1]
        self._dtype = x_train.dtype
        self._max_val = (torch.sign(torch.max(x_train)) * torch.ceil(torch.abs(torch.max(x_train)))).item()
        self._min_val = (torch.sign(torch.min(x_train)) * torch.ceil(torch.abs(torch.min(x_train)))).item()
        
        # Initialize membership function
        self._membership_function = membership_function()

        # Initialize premise parameters
        if init_mode == 0: # Based on data
            self._premises = Parameter(self._membership_function.initialize_premises(x_train=x_train, fuzzy_rules=init_fuzzy_rules), requires_grad=True)
        else: # Random initialization
            self._premises = Parameter(2 * torch.rand(self._input_size, init_fuzzy_rules, len(self._membership_function._params), dtype=self._dtype) - 1, requires_grad=True)



    def forward(self, x):
        return self._membership_function(x, self._premises)



    @property
    def premises_structure(self):
        df = pd.DataFrame()
        rules = ['Fuzzy rule {}'.format(i) for i in range(1, self._premises.data.shape[1]+1)]
        
        for i in range(self._input_size):
            for j in range(len(self._membership_function._params)):
                column = pd.Series(self._premises.data[i,:,j], index=rules, name=self._membership_function._params[j] + f' (x{i})', )
                df[self._membership_function._params[j] + f' (x{i})'] = column

        return df



    @property
    def plot_premises(self):
        variables = [f'x{i}' for i in range(self._input_size)]
        dataframe = self.premises_structure

        sep = round((0.1 * (self._max_val - self._min_val))) + 1
        x = np.linspace(self._min_val - sep, self._max_val + sep, 500)

        fig, axes = plt.subplots(nrows=self._premises.data.shape[1], ncols=len(variables), figsize=(15, 8), sharex=False, sharey=False)

        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        for i, rule in enumerate(dataframe.index):
            for j, var in enumerate(variables):
                params = [dataframe.loc[rule, f'{param} ({var})'] for param in self._membership_function._params]
                
                y = self._membership_function._simple_implementation(x, *params)

                ax = axes[i, j] if axes.ndim > 1 else axes[max(i, j)]
                ax.plot(x, y, label=f'{rule}, {var}')
                ax.set_title(f'{rule}, {var}')
                ax.grid(True)
                if i == self._premises.data.shape[1] - 1:
                    ax.set_xlabel('x')
                if j == 0:
                    ax.set_ylabel('Membership Value')

        plt.tight_layout()
        plt.show()