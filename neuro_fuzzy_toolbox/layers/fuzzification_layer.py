import torch
import torch.nn as nn
from torch.nn import Parameter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF

class FuzzificationLayer(nn.Module):
    
    def __init__(self, fuzzy_rules=1, membership_function=GeneralizedBell_MF, **kwargs):
        super(FuzzificationLayer, self).__init__()
        
        x_train = kwargs.get('x_train', None)
        input_size = kwargs.get('input_size', None)
        dtype = kwargs.get('dtype', torch.float32)
        
        if x_train is not None:
            input_size = x_train.shape[1]
            dtype = x_train.dtype
            premises = Parameter(membership_function().initialize_premises(x_train=x_train, fuzzy_rules=fuzzy_rules), requires_grad=True)
        elif input_size is None:
            raise ValueError("Must provide 'x_train' or 'input_size' to initialize the layer.")
        else:
            premises = Parameter(membership_function().random_premises(input_size, fuzzy_rules, dtype), requires_grad=True)
            
        
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        
        # Membership function
        self._membership_function = membership_function()

        # Initialize premise parameters
        self._premises = premises



    def forward(self, x):
        return self._membership_function(x, self._premises)
    
    
    
    def init_premises(self, x_train):
        self._dtype = x_train.dtype
        self._premises = Parameter(self._membership_function.initialize_premises(x_train=x_train, fuzzy_rules=self._premises.data.shape[1]), requires_grad=True)
        return



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

        x = np.linspace(self._membership_function._max_val_plot, self._membership_function._min_val_plot, 500)
        
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