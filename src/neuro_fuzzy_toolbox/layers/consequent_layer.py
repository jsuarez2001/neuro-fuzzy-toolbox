import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import pandas as pd

from neuro_fuzzy_toolbox.func import Linear_CF

class ConsequentLayer(nn.Module):
    """
    Consequent layer for an Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Computes the weighted output of each fuzzy rule using a linear consequent
    function. Consequent parameters are stored as a single trainable tensor.
    """
    def __init__(self, input_size, rules, outputs=1, features=None, dtype=torch.float32):
        """
        Initializes a new ConsequentLayer instance.

        Args:
            input_size (int): Number of input features of the ANFIS model.
            rules (int): Number of fuzzy rules in the ANFIS model.
            outputs (int): Number of model outputs. Defaults to ``1``.
            features (iterable, optional): Names of the input features as
                strings, of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type for the layer parameters.
                Defaults to ``torch.float32``.
        """
        super(ConsequentLayer, self).__init__()
        # Input data info
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
            
        # Initialize consequent function
        self._consequent_function = Linear_CF()
            
        self._consequents = Parameter(self._consequent_function.random_consequents(outputs=outputs,
                                                                                       rules=rules, 
                                                                                       input_size=input_size, 
                                                                                       dtype=dtype), requires_grad=True)
        
    
    def forward(self, x, weights):
        """
        Forward pass of the consequent layer.

        Computes the weighted output of each rule using the normalized firing
        levels as weights.

        Args:
            x (torch.Tensor): Input features of shape
                ``(batch_size, input_size)``.
            weights (torch.Tensor): Normalized firing levels of shape
                ``(batch_size, rules)``.

        Returns:
            torch.Tensor: Weighted rule outputs of shape
            ``(outputs, batch_size, rules)``.
        """
        return self._consequent_function(x, self._consequents, weights)
    
    
    def get_consequents_outputs(self, x):
        """
        Returns the individual rule outputs without weighting by normalized
        firing levels.

        Args:
            x (torch.Tensor): Input features of shape
                ``(batch_size, input_size)``.

        Returns:
            torch.Tensor: Unweighted rule outputs of shape
            ``(outputs, batch_size, rules)``.
        """
        return self._consequent_function.get_consequents_outputs(x, self._consequents)

    
    @property
    def get_consequents_structure(self):
        """
        Structure of the consequent parameters.

        Returns:
            list[pd.DataFrame]: List of DataFrames, one per model output,
            each describing the consequent parameters for every rule and
            input feature.
        """
        consequents_tensor = self.get_consequents()
        dfs = []
        num_outputs = consequents_tensor.shape[0]
        num_rules = consequents_tensor.shape[1]
        num_params = consequents_tensor.shape[2]

        rules = [f'rule {i}' for i in range(1, num_rules + 1)]

        for o in range(num_outputs):
            data = []
            column_tuples = []

            for i in range(num_params):
                input_var = self.features[i] if i < num_params - 1 else ""
                c_name = f'c{i}'

                column_tuples.append((input_var, c_name))

                data.append(consequents_tensor[o, :, i].cpu().numpy())

            df = pd.DataFrame(
                data=np.array(data).T,
                index=rules,
                columns=pd.MultiIndex.from_tuples(
                    tuples=column_tuples
                )
            )
            dfs.append(df)

        return dfs
    
    
    def _to_dtype(self, dtype):
        """
        Casts the consequent parameters to a different data type.

        Args:
            dtype (torch.dtype): Target data type.
        """
        self._consequents.data = self._consequents.data.type(dtype)
        
        
    def get_consequents(self):
        """
        Returns the current consequent parameters.

        Returns:
            torch.Tensor: Consequent parameter tensor of shape
            ``(outputs, rules, input_size + 1)``.
        """
        return self._consequents.data.clone().detach()
    
    
    def set_consequents(self, consequents):
        """
        Sets the consequent parameters.

        Args:
            consequents (torch.Tensor): Consequent parameter tensor of shape
                ``(outputs, rules, input_size + 1)``.
        """
        self._consequents = Parameter(consequents, requires_grad=True)
    
    
    def get_consequents_as_parameters_list(self):
        """
        Returns the consequent parameters as a single-element list of trainable parameters, useful for passing to optimizers.

        Returns:
            list[nn.Parameter]: List containing a single ``nn.Parameter`` with all consequent parameters.
        """
        return [self._consequents]
        


class alt_ConsequentLayer(nn.Module):
    """
    Alternative consequent layer for an Adaptive Neuro-Fuzzy Inference System (ANFIS).
    
    Functionally equivalent to :class:`ConsequentLayer`, but stores the consequent parameters as a :class:`nn.ParameterList` 
    of per-rule tensors rather than a single tensor. This representation is required by optimizers and training algorithms 
    that add or remove rules dynamically during structural adaptation.
    """
    def __init__(self, input_size, rules, outputs=1, features=None, dtype=torch.float32):
        """
        Initializes a new alt_ConsequentLayer instance.

        Args:
            input_size (int): Number of input features of the ANFIS model.
            rules (int): Number of fuzzy rules in the ANFIS model.
            outputs (int): Number of model outputs. Defaults to ``1``.
            features (iterable, optional): Names of the input features as strings, of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type for the layer parameters. Defaults to ``torch.float32``.
        """
        super(alt_ConsequentLayer, self).__init__()
        # Input data info
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
        
        # Initialize consequent function
        self._consequent_function = Linear_CF()
            
        self._consequents = nn.ParameterList([
            nn.Parameter(consequent, requires_grad=True) for consequent in self._consequent_function.random_consequents(
                outputs=outputs,
                rules=rules, 
                input_size=input_size, 
                dtype=dtype
            ).unbind(1)
        ])
        
        
    def forward(self, x, weights):
        """
        Forward pass of the alternative consequent layer.

        Computes the weighted output of each rule using the normalized firing
        levels as weights.

        Args:
            x (torch.Tensor): Input features of shape ``(batch_size, input_size)``.
            weights (torch.Tensor): Normalized firing levels of shape ``(batch_size, rules)``.

        Returns:
            torch.Tensor: Weighted rule outputs of shape ``(outputs, batch_size, rules)``.
        """
        return self._consequent_function(x, torch.stack([consequent for consequent in self._consequents], 1), weights)
    
    
    def get_consequents_outputs(self, x):
        """
        Returns the individual rule outputs without weighting by normalized firing levels.

        Args:
            x (torch.Tensor): Input features of shape ``(batch_size, input_size)``.

        Returns:
            torch.Tensor: Unweighted rule outputs of shape ``(outputs, batch_size, rules)``.
        """
        return self._consequent_function.get_consequents_outputs(x, torch.stack([consequent for consequent in self._consequents], 1))
    
    
    @property
    def get_consequents_structure(self):
        """
        Structure of the consequent parameters.

        Returns:
            list[pd.DataFrame]: List of DataFrames, one per model output, each describing the consequent parameters for every rule
            and input feature.
        """
        consequents_tensor = torch.stack(
            [consequent.data.clone().detach() for consequent in self._consequents], 1
        )
        dfs = []
        num_outputs = consequents_tensor.shape[0]
        num_rules = consequents_tensor.shape[1]
        num_params = consequents_tensor.shape[2]

        rules = [f'rule {i}' for i in range(1, num_rules + 1)]

        for o in range(num_outputs):
            data = []
            column_tuples = []

            for i in range(num_params):
                input_var = self.features[i] if i < num_params - 1 else ""
                c_name = f'c{i}'

                column_tuples.append((input_var, c_name))

                data.append(consequents_tensor[o, :, i].cpu().numpy())

            df = pd.DataFrame(
                data=np.array(data).T,
                index=rules,
                columns=pd.MultiIndex.from_tuples(
                    tuples=column_tuples
                )
            )
            dfs.append(df)

        return dfs
    
    
    def _to_dtype(self, dtype):
        """
        Casts the consequent parameters to a different data type.

        Args:
            dtype (torch.dtype): Target data type.
        """
        for consequent in self._consequents:
            consequent.data = consequent.data.type(dtype)
            
        
    def get_consequents(self):
        """
        Returns the current consequent parameters as a single tensor.

        Returns:
            torch.Tensor: Consequent parameter tensor of shape ``(outputs, rules, input_size + 1)``.
        """
        return torch.stack([consequent.data.clone().detach() for consequent in self._consequents], 1)
    
    
    def set_consequents(self, consequents):
        """
        Sets the consequent parameters from a single tensor, converting each rule's parameters into a separate trainable
        entry in the internal :class:`nn.ParameterList`.

        Args:
            consequents (torch.Tensor): Consequent parameter tensor of shape ``(outputs, rules, input_size + 1)``.
        """
        self._consequents = nn.ParameterList([
            nn.Parameter(consequent) for consequent in consequents.unbind(1)
        ])
        
    
    def get_consequents_as_parameters_list(self):
        """
        Returns the consequent parameters as a PyTorch ParameterList, useful for passing to optimizers.

        Returns:
            nn.ParameterList: List of per-rule consequent parameters.
        """
        return self._consequents