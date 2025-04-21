import torch
import torch.nn as nn
from torch.nn import Parameter

from EXP_neuro_fuzzy_toolbox.func import (
    GeneralizedBell_MF
)

from EXP_neuro_fuzzy_toolbox.models import (
    h_AntecedentBlock,
    AntecedentBlock
)

JOIN_MODES = ["cat", "bilinear", "element-wise"]

class cat_joiner(torch.nn.Module):
    def forward(self, x, w):
        return torch.cat((x, w), dim=1)
    
class bilinear_joiner(torch.nn.Module):
    def __init__(self, input_size, num_rules, bilinear_outputs):
        super(bilinear_joiner, self).__init__()
        self.layer = nn.Bilinear(input_size, num_rules, bilinear_outputs)
        
    def forward(self, x, w):
        return self.layer(x, w)
    
class element_wise_joiner(torch.nn.Module):
    def forward(self, x, w):
        return torch.repeat_interleave(x, w.shape[1], dim=1) * w.repeat(1, x.shape[1])



class h_DeepANFIS(torch.nn.Module):
    def __init__(self, input_size, num_mfs, consequent_model, outputs=1, normalization=True, rule_reduced=False, membership_function=GeneralizedBell_MF, mode="cat", bilinear_outputs=None, output_type="regression", dtype=torch.float32):
        super(h_DeepANFIS, self).__init__()
        
        if mode not in JOIN_MODES:
            raise ValueError(f"Mode must be one of {JOIN_MODES}")
        if mode == "bilinear" and bilinear_outputs is None:
            raise ValueError("bilinear_outputs must be provided when mode is bilinear")
        if mode != "bilinear" and bilinear_outputs is not None:
            raise ValueError("bilinear_outputs must be None when mode is not bilinear")
        
        antecedent_block = h_AntecedentBlock(
            input_size=input_size,
            num_mfs=num_mfs,
            normalization=normalization,
            rule_reduced=rule_reduced,
            membership_function=membership_function,
            dtype=dtype
        )
        
        # Information from the antescedent block
        self._input_size = antecedent_block._input_size
        self._normalization = antecedent_block._normalization
        self._membership_function = antecedent_block._membership_function
        self._dtype = antecedent_block._dtype
        
        # Model Information
        self._outputs = outputs
        self._output_type = output_type
        
        # Blocks
        self._antecedent_block = antecedent_block
        
        self._firing_levels_input_joiner = None
        self._consequent_model = None

        if mode == "cat":
            self._firing_levels_input_joiner = cat_joiner()
            self._consequent_model = consequent_model(input_size + self.rules, outputs, dtype)
        elif mode == "bilinear":
            self._firing_levels_input_joiner = bilinear_joiner(input_size, self.rules, bilinear_outputs)
            self._consequent_model = consequent_model(bilinear_outputs, outputs, dtype)
        elif mode == "element-wise":
            self._firing_levels_input_joiner = element_wise_joiner()
            self._consequent_model = consequent_model(input_size * self.rules, outputs, dtype)
            
    def init_premises(self, x_train):
        self._antecedent_block.init_premises(x_train)

    def forward(self, x):
        # Antecedent block
        w = self._antecedent_block(x)
        
        # Joiner
        output = self._firing_levels_input_joiner(x, w)
        
        # Consequent model
        output = self._consequent_model(output)
        
        return output
    
    @property
    def rules(self):
        return self._antecedent_block.rules
    
    def plot_premises(self, mf=None, input_dim=None, grouped_by_dim=False):
        self._antecedent_block.plot_premises(mf, input_dim, grouped_by_dim)
        
    def get_premises_as_parameters_list(self):
        return self._antecedent_block.get_premises_as_parameters_list()
    
    def get_consequents_as_parameters_list(self):
        return self._consequent_model.get_consequents_as_parameters_list()
    
    
class DeepANFIS(torch.nn.Module):
    def __init__(self, mf_distribution, consequent_model, outputs, normalization=True, membership_function=GeneralizedBell_MF, mode="cat", bilinear_outputs=None, dtype=torch.float32):
        super(DeepANFIS, self).__init__()
        
        if mode not in JOIN_MODES:
            raise ValueError(f"Mode must be one of {JOIN_MODES}")
        if mode == "bilinear" and bilinear_outputs is None:
            raise ValueError("bilinear_outputs must be provided when mode is bilinear")
        if mode != "bilinear" and bilinear_outputs is not None:
            raise ValueError("bilinear_outputs must be None when mode is not bilinear")
        
        antecedent_block = AntecedentBlock(mf_distribution, normalization, membership_function, dtype)
        
        # Information from the antescedent block
        self._input_size = antecedent_block._input_size
        self._normalization = antecedent_block._normalization
        self._membership_function = antecedent_block._membership_function
        self._dtype = antecedent_block._dtype
        
        # Blocks
        self._antecedent_block = antecedent_block
        
        self._firing_levels_input_joiner = None
        self._consequent_model = None
        if mode == "cat":
            self._firing_levels_input_joiner = cat_joiner()
            self._consequent_model = consequent_model(self._input_size + self.rules, outputs)
        elif mode == "bilinear":
            self._firing_levels_input_joiner = bilinear_joiner(self._input_size, self.rules, bilinear_outputs)
            self._consequent_model = consequent_model(bilinear_outputs, outputs)
        elif mode == "element-wise":
            self._firing_levels_input_joiner = element_wise_joiner()
            self._consequent_model = consequent_model(self._input_size * self.rules, outputs)
            
            
    def init_premises(self, x_train):
        self._antecedent_block.init_premises(x_train)
            
    
    def forward(self, x):
        # Antecedent block
        w = self._antecedent_block(x)
        
        # Joiner
        output = self._firing_levels_input_joiner(x, w)
        
        # Consequent model
        output = self._consequent_model(output)
        
        return output
    
    @property
    def rules(self):
        return self._antecedent_block.rules
    
    def plot_premises(self, mf=None, input_dim=None, grouped_by_dim=False):
        self._antecedent_block.plot_premises(mf, input_dim, grouped_by_dim)
        
    def get_premises_as_parameters_list(self):
        return self._antecedent_block.get_premises_as_parameters_list()
    
    def get_consequents_as_parameters_list(self):
        return self._consequent_model.get_consequents_as_parameters_list()