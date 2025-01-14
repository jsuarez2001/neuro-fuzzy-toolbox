import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF, Linear_CF
from neuro_fuzzy_toolbox.layers import FuzzificationLayer, FiringLevelsLayer, NormalizationLayer, ConsequentLayer, OutputLayer

class baseANFIS(nn.Module):
    def __init__(self):
        super(baseANFIS, self).__init__()
        
        # Input data info
        self._input_size = None
        self._dtype = None
        
        # ANFIS info
        self._rule_reduced = None
        self._outputs = None
        self._output_type = None
        
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
    
    
    
    def init_premises(self, x_train):
        self._dtype = x_train.dtype
        self._consequent_layer._to_dtype(x_train.dtype) # Set dtype to consequents
        self._fuzzification_layer.init_premises(x_train)
        return
    
    
    
    # ---- Model predict ----
    def predict(self, x):
        output = self.forward(x).detach().numpy()
        
        if self._output_type == 'multiclass':
            output = nn.Softmax(dim=1)(self.forward(x)).detach().numpy()
            output = np.argmax(output, axis=1, keepdims=True)
            
        elif self._output_type == 'binary':
            output = (output > 0.5).astype(int)
            
        return output
    

    # ---- Intermediate values ----
    def intermediate_values(self, x):
        with torch.no_grad():
            w = self._fuzzification_layer(x)
            w = self._firing_levels_layer(w)
            w_norm = self._normalization_layer(w)
            outputs = self._consequent_layer(x, w_norm)
        return w, w_norm, outputs
    
    
    # ----- Parameter seters and getters -----
    def set_premises(self, premises):
        self._fuzzification_layer._premises = Parameter(premises, requires_grad=True)
        
    def set_consequents(self, consequents):
        self._consequent_layer._consequents = Parameter(consequents, requires_grad=True)
        
    def get_premises(self):
        return self._fuzzification_layer._premises.data.clone()
    
    def get_consequents(self):
        return self._consequent_layer._consequents.data.clone()
    
    
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
    def plot_premises(self):
        self._fuzzification_layer.plot_premises



class ANFIS(baseANFIS):

    def __init__(self, fuzzy_rules=1, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, output_type="regression", rule_reduced=False, **kwargs):
        super(ANFIS, self).__init__()
        
        x_train = kwargs.get('x_train', None)
        input_size = kwargs.get('input_size', None)
        dtype = kwargs.get('dtype', torch.float32)
        
        if x_train is not None:
            if input_size is not None:
                raise ValueError("Provide either 'x_train' or 'input_size' but not both.")
            input_size = x_train.shape[1]
            dtype = x_train.dtype
        elif input_size is None:
            raise ValueError("Must provide 'x_train' or 'input_size' to initialize the model.")
        
        
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        
        
        # ANFIS info
        self._rule_reduced = rule_reduced
        self._output_type = output_type
        self._outputs = outputs
        
        
        # Layers
        self._fuzzification_layer = FuzzificationLayer(
            x_train=x_train,
            input_size=input_size,
            dtype=dtype,
            fuzzy_rules=fuzzy_rules, 
            membership_function=membership_function, 
            )
        
        self._firing_levels_layer = FiringLevelsLayer(rule_reduced=rule_reduced)
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = ConsequentLayer(
            input_size=self._input_size, 
            dtype=self._dtype, 
            fuzzy_rules=fuzzy_rules, 
            consequent_function=consequent_function, 
            outputs=outputs,
            rule_reduced=rule_reduced
            )
        
        self._output_layer = OutputLayer(output_type=self._output_type)