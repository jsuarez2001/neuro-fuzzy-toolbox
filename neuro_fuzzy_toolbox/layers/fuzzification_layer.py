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
        rules = ['Fuzzy rule {}'.format(i) for i in range(0, self._premises.data.shape[1])]
        
        for i in range(self._input_size):
            for j in range(len(self._membership_function._params)):
                column = pd.Series(self._premises.data[i,:,j], index=rules, name=self._membership_function._params[j] + f' (x{i})', )
                df[self._membership_function._params[j] + f' (x{i})'] = column

        return df
    
    
    
    def plot_premises(self, fuzzy_rule=None, input_dim=None):
        variables = [f'x{i}' for i in range(self._input_size)]
        dataframe = self.premises_structure

        x = np.linspace(self._membership_function._max_val_plot, self._membership_function._min_val_plot, 500)

        # Determine which rules and dimensions to plot
        if fuzzy_rule is not None:
            # Convert numeric index to string format if necessary
            if isinstance(fuzzy_rule, (int, float)):
                fuzzy_rule = f'Fuzzy rule {fuzzy_rule}'
            # Validate that the rule exists
            if fuzzy_rule not in dataframe.index:
                raise ValueError(f"Fuzzy rule '{fuzzy_rule}' not found in premises. Available rules: {dataframe.index.tolist()}")
            rules_to_plot = [fuzzy_rule]
        else:
            rules_to_plot = dataframe.index

        # Validate input dimension
        if input_dim is not None:
            if not isinstance(input_dim, int) or input_dim < 0 or input_dim >= len(variables):
                raise ValueError(f"input_dim must be between 0 and {len(variables)-1}")
            dims_to_plot = [input_dim]
        else:
            dims_to_plot = range(len(variables))

        # Calculate subplot dimensions
        n_rules = len(rules_to_plot)
        n_dims = len(dims_to_plot)

        # Create subplots based on the number of rules and dimensions
        if n_rules == 1 and n_dims == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            axes = np.array([[ax]])
        else:
            fig, axes = plt.subplots(nrows=n_rules, ncols=n_dims, figsize=(5*n_dims, 4*n_rules), squeeze=False)

        for i, rule in enumerate(rules_to_plot):
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                try:
                    params = [dataframe.loc[rule, f'{param} ({var})'] for param in self._membership_function._params]

                    y = self._membership_function._simple_numpy_implementation(x, *params)

                    ax = axes[i, j]
                    ax.plot(x, y, label=f'{rule}, {var}')
                    ax.set_title(f'{rule}, {var}')
                    ax.grid(True)
                    if i == n_rules - 1:
                        ax.set_xlabel('x')
                    if j == 0:
                        ax.set_ylabel('Membership Value')
                except KeyError as e:
                    print(f"Warning: Could not find parameters for rule '{rule}' and variable '{var}'")
                    continue
                
        plt.tight_layout()
        plt.show()