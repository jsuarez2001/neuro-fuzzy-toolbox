import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF, Linear_CF
from neuro_fuzzy_toolbox.layers import FuzzificationLayer, FiringLevelsLayer, NormalizationLayer, ConsequentLayer, OutputLayer
from neuro_fuzzy_toolbox.layers import rule_reduced_FiringLevelsLayer, complete_FiringLevelsLayer
from neuro_fuzzy_toolbox.layers import rule_reduced_ConsequentLayer, complete_ConsequentLayer

class baseANFIS(nn.Module):
    def __init__(self):
        super(baseANFIS, self).__init__()
        
        # Input data info
        self._input_size = None
        self._input_dtype = None
        
        # ANFIS info
        self._rule_reduced = None
        self._outputs = None
        
        # Layers
        self._fuzzification_layer = None
        self._firing_levels_layer = None
        self._normalization_layer = None
        self._consequent_layer = None
        self._output_layer = None


    # ---- Forward pass ----
    def forward(self, x):
        output = self._fuzzification_layer(x)
        output = self._consequent_layer(x, self._normalization_layer(self._firing_levels_layer(output)))
        output = self._output_layer(output)
        return output
    
    
    
    # ---- Model predict ----
    def predict(self, x):
        output = self.forward(x).detach().numpy()
        
        if isinstance(self._output_layer._last_layer, nn.Softmax):
            output = np.argmax(output, axis=1, keepdims=True)
            
        elif isinstance(self._output_layer._last_layer, nn.Sigmoid):
            output = (output > 0.5).astype(int)
            
        return output
    

    # ---- Intermediate values ----
    def intermediate_values(self, x):
        with torch.no_grad():
            w = self._fuzzification_layer(x)
            w = self._firing_levels_layer(w)
            w_norm = self._normalization_layer(w)
        return w, w_norm
    
    
    # ----- Parameter seters and getters -----
    def set_premises(self, premises):
        self._fuzzification_layer._premises = Parameter(premises, requires_grad=True)
        
    def set_consequents(self, consequents):
        self._consequent_layer._consequents = Parameter(consequents, requires_grad=True)
        
    def get_premises(self):
        return self._fuzzification_layer._premises.data
    
    def get_consequents(self):
        return self._consequent_layer._consequents.data
    
    
    # ---- ANFIS parameters info ----
    @property
    def fuzzy_rules(self):
        return self.get_premises().shape[1]
    
    @property
    def rules(self):
        return self.get_consequents().shape[1]
    
    
    # ----- Parameters dataframes -----
    @property
    def premises_structure(self):
        return self._fuzzification_layer.premises_structure
    
    @property
    def consequents_structure(self):
        return self._consequent_layer.consequents_structure
    
    def show_premises_structure(self):
        print(self.premises_structure)
        
    def show_consequents_structure(self):
        output = 1
        for df in self.consequents_structure:
            print(f'- Output {output}:')
            print(df)
            print('\n')
            output += 1
    
    
    # ----- Plot premises -----
    @property
    def plot_premises(self):
        self._fuzzification_layer.plot_premises



class ANFIS(baseANFIS):

    def __init__(self, x_train, init_fuzzy_rules=1, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, premises_init_mode=0, output_type=None, rule_reduced=False):
        super(ANFIS, self).__init__()
        
        # Input data info
        self._input_size = x_train.shape[1]
        self._input_dtype = x_train.dtype
        
        
        # ANFIS info
        self._rule_reduced = rule_reduced
        self._outputs = outputs
        
        
        # Layers
        self._fuzzification_layer = FuzzificationLayer(x_train=x_train,
                                           init_fuzzy_rules=init_fuzzy_rules, 
                                           membership_function=membership_function, 
                                           init_mode=premises_init_mode)
        
        self._firing_levels_layer = FiringLevelsLayer(rule_reduced=rule_reduced)
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = ConsequentLayer(input_size=self._input_size, 
                                                 dtype=self._input_dtype, 
                                                 init_fuzzy_rules=init_fuzzy_rules, 
                                                 consequent_function=consequent_function, 
                                                 outputs=outputs,
                                                 rule_reduced=rule_reduced)
        
        self._output_layer = OutputLayer(output_type=output_type)



class rule_reduced_ANFIS(baseANFIS):
    
    def __init__(self, x_train, init_fuzzy_rules=1, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, premises_init_mode=0, output_type=None):
        super(rule_reduced_ANFIS, self).__init__()
        
        # Input data info
        self._input_size = x_train.shape[1]
        self._input_dtype = x_train.dtype
        
        
        # ANFIS info
        self._outputs = outputs
        
        
        # Layers
        self._fuzzification_layer = FuzzificationLayer(x_train=x_train,
                                           init_fuzzy_rules=init_fuzzy_rules, 
                                           membership_function=membership_function, 
                                           init_mode=premises_init_mode)
        
        self._firing_levels_layer = rule_reduced_FiringLevelsLayer()
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = rule_reduced_ConsequentLayer(input_size=self._input_size, 
                                                 dtype=self._input_dtype, 
                                                 init_fuzzy_rules=init_fuzzy_rules, 
                                                 consequent_function=consequent_function, 
                                                 outputs=outputs)
        
        self._output_layer = OutputLayer(output_type=output_type)



class complete_ANFIS(baseANFIS):
    
    def __init__(self, x_train, init_fuzzy_rules=1, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, premises_init_mode=0, output_type=None):
        super(complete_ANFIS, self).__init__()
        
        # Input data info
        self._input_size = x_train.shape[1]
        self._input_dtype = x_train.dtype
        
        
        # ANFIS info
        self._outputs = outputs
        
        
        # Layers
        self._fuzzification_layer = FuzzificationLayer(x_train=x_train,
                                           init_fuzzy_rules=init_fuzzy_rules, 
                                           membership_function=membership_function, 
                                           init_mode=premises_init_mode)
        
        self._firing_levels_layer = complete_FiringLevelsLayer()
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = complete_ConsequentLayer(input_size=self._input_size, 
                                                 dtype=self._input_dtype, 
                                                 init_fuzzy_rules=init_fuzzy_rules, 
                                                 consequent_function=consequent_function, 
                                                 outputs=outputs)
        
        self._output_layer = OutputLayer(output_type=output_type)



