import torch
import torch.nn as nn
from torch.nn import Parameter
import pandas as pd

from neuro_fuzzy_toolbox.func import Linear_CF

class ConsequentLayer(nn.Module):
    """
    Clase para representar la capa consecuente de un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS).
    """
    def __init__(self, input_size, rules, outputs=1, consequent_function=Linear_CF, dtype=torch.float32):
        """
        Inicializa una nueva instancia de la clase ConsequentLayer.
        
        Args:
            input_size (int): Número de features de los datos de entrada (del modelo ANFIS, no de la capa).
            rules (int): Número de reglas del modelo ANFIS, depende del modelo en sí.
            outputs (int): Número de salidas del modelo ANFIS (Default: 1).
            consequent_function (ConsequentFunction): Función consecuente a utilizar en la capa (Default: Linear_CF).
            dtype (torch.dtype): Tipo de dato a utilizar en el modelo (Default: torch.float32).
        """
        super(ConsequentLayer, self).__init__()
        # Initialize consequent function
        self._consequent_function = consequent_function()
            
        self._consequents = Parameter(self._consequent_function.random_consequents(outputs=outputs,
                                                                                       rules=rules, 
                                                                                       input_size=input_size, 
                                                                                       dtype=dtype), requires_grad=True)
        
    
    def forward(self, x, weights):
        """
        Realiza un paso hacia adelante para calcular la salida de la capa consecuente.
        
        Args:
            x (torch.Tensor): Tensor de tamaño (batch_size, input_size) que contiene los features de entrada.
            weights (torch.Tensor): Tensor de tamaño (batch_size, rules) que contiene los pesos de las reglas.
            
        Returns:
            torch.Tensor: Tensor de tamaño (outputs, batch_size, rules) que contiene las salidas del modelo ANFIS.
        """
        return self._consequent_function(x, self._consequents, weights)
    
    
    @property
    def consequents_structure(self):
        """
        Retorna la estructura de los parámetros consecuentes.
        
        Returns:
            list: Lista de DataFrames de pandas que contienen la estructura de los parámetros consecuentes
        """
        dfs = [pd.DataFrame() for _ in range(self._consequents.data.shape[0])]

        rules = ['rule {}'.format(i) for i in range(1, self._consequents.data.shape[1]+1)]

        for o in range(self._consequents.data.shape[0]):
            for i in range(self._consequents.data.shape[2]):
                name=f'c{i} (x{i})'
                if (i == self._consequents.data.shape[2]-1): name=f'c{i}'
                column = pd.Series(self._consequents.data[o,:,i], index=rules, name=name)
                dfs[o][name] = column

        return dfs
    
    
    def _to_dtype(self, dtype):
        """
        Cambia el tipo de dato de los parámetros consecuentes.
        
        Args:
            dtype (torch.dtype): Tipo de dato al que se desea cambiar.
        """
        self._consequents.data = self._consequents.data.type(dtype)
        
        
    def get_consequents(self):
        """
        Retorna los parámetros consecuentes del modelo.
        
        Returns:
            torch.tensor: Tensor con los parámetros consecuentes. Su forma es (outputs, rules, input_size + 1).
        """
        return self._consequents.data.clone().detach()
    
    
    def set_consequents(self, consequents):
        """
        Setea los parámetros consecuentes del modelo.
        
        Args:
            consequents (torch.tensor): Tensor con los parámetros consecuentes. Su forma debe ser (outputs, rules, input_size + 1).
        """
        self._consequents = Parameter(consequents, requires_grad=True)
    
    
    def get_consequents_as_parameters_list(self):
        """
        Retorna los parámetros consecuentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            list: Lista de 1 solo elemento (nn.Parameter) que contiene los parámetros consecuentes.
        """
        return [self._consequents]
        


class alt_ConsequentLayer(nn.Module):
    """
    Clase para representar la capa consecuente de un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS). Tiene la particularidad de las reglas se almacenan en una lista de tensores y no en un tensor único.
    """
    def __init__(self, input_size, rules, outputs=1, consequent_function=Linear_CF, dtype=torch.float32):
        """
        Inicializa una nueva instancia de la clase alt_ConsequentLayer.
        
        Args:
            input_size (int): Número de features de los datos de entrada (del modelo ANFIS, no de la capa).
            rules (int): Número de reglas del modelo ANFIS, depende del modelo en sí.
            outputs (int): Número de salidas del modelo ANFIS (Default: 1).
            consequent_function (ConsequentFunction): Función consecuente a utilizar en la capa (Default: Linear_CF).
            dtype (torch.dtype): Tipo de dato a utilizar en el modelo (Default: torch.float32).
        """
        super(alt_ConsequentLayer, self).__init__()
        # Initialize consequent function
        self._consequent_function = consequent_function()
            
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
        Realiza un paso hacia adelante para calcular la salida de la capa consecuente.
        
        Args:
            x (torch.Tensor): Tensor de tamaño (batch_size, input_size) que contiene los features de entrada.
            weights (torch.Tensor): Tensor de tamaño (batch_size, rules) que contiene los pesos de las reglas.
            
        Returns:
            torch.Tensor: Tensor de tamaño (outputs, batch_size, rules) que contiene las salidas del modelo ANFIS.
        """
        return self._consequent_function(x, torch.stack([consequent for consequent in self._consequents], 1), weights)
    
    
    @property
    def consequents_structure(self):
        """
        Retorna la estructura de los parámetros consecuentes.
        
        Returns:
            list: Lista de DataFrames de pandas que contienen la estructura de los parámetros consecuentes
        """
        consequents_tensor = torch.stack([consequent.data.clone().detach() for consequent in self._consequents], 1)
        dfs = [pd.DataFrame() for _ in range(consequents_tensor.shape[0])]

        rules = ['rule {}'.format(i) for i in range(1, consequents_tensor.shape[1]+1)]

        for o in range(consequents_tensor.shape[0]):
            for i in range(consequents_tensor.shape[2]):
                name=f'c{i} (x{i})'
                if (i == consequents_tensor.shape[2]-1): name=f'c{i}'
                column = pd.Series(consequents_tensor[o,:,i], index=rules, name=name)
                dfs[o][name] = column

        return dfs
    
    
    def _to_dtype(self, dtype):
        """
        Cambia el tipo de dato de los parámetros consecuentes.
        
        Args:
            dtype (torch.dtype): Tipo de dato al que se desea cambiar.
        """
        for consequent in self._consequents:
            consequent.data = consequent.data.type(dtype)
            
        
    def get_consequents(self):
        """
        Retorna los parámetros consecuentes del modelo.
        
        Returns:
            torch.tensor: Tensor con los parámetros consecuentes. Su forma es (outputs, rules, input_size + 1).
        """
        return torch.stack([consequent.data.clone().detach() for consequent in self._consequents], 1)
    
    
    def set_consequents(self, consequents):
        """
        Setea los parámetros consecuentes del modelo.
        
        Args:
            consequents (torch.tensor): Tensor con los parámetros consecuentes. Su forma debe ser (outputs, rules, input_size + 1).
        """
        self._consequents = nn.ParameterList([
            nn.Parameter(consequent) for consequent in consequents.unbind(1)
        ])
        
    
    def get_consequents_as_parameters_list(self):
        """
        Retorna los parámetros consecuentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            nn.ParameterList: Lista de parámetros consecuentes.
        """
        return self._consequents