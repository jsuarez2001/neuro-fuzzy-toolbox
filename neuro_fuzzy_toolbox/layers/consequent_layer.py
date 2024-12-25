import torch.nn as nn
from torch.nn import Parameter
import pandas as pd

from neuro_fuzzy_toolbox.func import Linear_CF

class ConsequentLayer(nn.Module):
    
    def __init__(self, input_size, dtype, init_fuzzy_rules, consequent_function=Linear_CF, outputs=1, rule_reduced=False):
        super(ConsequentLayer, self).__init__()
        # Initialize consequent function
        self._consequent_function = consequent_function()
        
        # Initialize consequent parameters
        if rule_reduced:
            init_consequents_rules = init_fuzzy_rules
        else:
            init_consequents_rules = init_fuzzy_rules**input_size
            
        self._consequents = Parameter(self._consequent_function.initialize_consequents(outputs=outputs,
                                                                                       consequents_rules=init_consequents_rules, 
                                                                                       input_size=input_size, 
                                                                                       dtype=dtype), requires_grad=True)
        
    
    def forward(self, x, weights):
        return self._consequent_function(x, self._consequents, weights)
    
    
    @property
    def consequents_structure(self):
        dfs = [pd.DataFrame() for _ in range(self._consequents.data.shape[0])]

        rules = ['rule {}'.format(i) for i in range(1, self._consequents.data.shape[1]+1)]

        for o in range(self._consequents.data.shape[0]):
            for i in range(self._consequents.data.shape[2]):
                name=f'c{i} (x{i})'
                if (i == self._consequents.data.shape[2]-1): name=f'c{i}'
                column = pd.Series(self._consequents.data[o,:,i], index=rules, name=name)
                dfs[o][name] = column

        return dfs