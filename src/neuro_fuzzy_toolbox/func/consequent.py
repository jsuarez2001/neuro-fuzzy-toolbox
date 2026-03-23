import torch
import torch.nn as nn

from abc import abstractmethod

class ConsequentFunction(nn.Module):
    """
    Clase abstracta para funciones consecuentes, solo define la estructura necesaria para diseñarlas en el toolbox. Sirve de guía para futuras implementaciones.
    """
    @abstractmethod
    def forward(self, x, consequents, weights):
        pass
    
    @abstractmethod
    def get_consequents_outputs(self, x, consequents):
        pass
    
    @abstractmethod
    def random_consequents(self, outputs, rules, input_size, dtype):
        pass


class Linear_CF(ConsequentFunction):
    """
    Función consecuente lineal, se encarga de calcular la salida de la red neuro-difusa a partir de los antecedentes y los pesos de las reglas bajo la forma de una combinación lineal.
    
    Esta función consecuente se define como:
    
    .. math::
    
        O_j = \\sum_{i=1}^{n} (c_{i,j} * x_i) + c_{n+1,j}
        
    donde:
        - :math:`O_j` es la j-ésima salida de la capa consecuente de un modelo ANFIS.
        - :math:`x_i` es el feature :math:`i` de un dato de entrada :math:`x` (que es de tamaño n).
        - :math:`c_{i,j}` es el i-ésimo parámetro consecuente asociado a la j-ésima regla de un modelo ANFIS.

    """
    def forward(self, x, consequents, weights):
        """
        Paso hacia adelante de la función consecuente lineal.
        
        Args:
            x (torch.tensor): Tensor de tamaño (batch_size, input_size) que contiene los features de entrada.
            consequents (torch.tensor): Tensor de tamaño (outputs, rules, input_size + 1) que contiene los parámetros consecuentes de la red ANFIS, donde *rules* es el número de reglas y *outputs* es el número de salidas del modelo.
            weights (torch.tensor): Tensor de tamaño (batch_size, rules) que contiene los pesos de las reglas.
        
        Returns:
            torch.tensor: Tensor de tamaño (outputs, batch_size, rules) que contiene las salidas de la red ANFIS.
        
        """
        return (torch.bmm(x.unsqueeze(0).expand(consequents[:, :, :-1].size(0), -1, -1), torch.transpose(consequents[:, :, :-1], 1, 2)) + consequents[:, :, -1].unsqueeze(1)).mul(weights.unsqueeze(0))
    
    def get_consequents_outputs(self, x, consequents):
        """
        Retorna las salidas de las reglas sin ponderación (sin multiplicar oor los normalized firing levels)
        
        Args:
            x (torch.tensor): Tensor de tamaño (batch_size, input_size) que contiene los features de entrada.
            consequents (torch.tensor): Tensor de tamaño (outputs, rules, input_size + 1) que contiene los parámetros consecuentes de la red ANFIS, donde *rules* es el número de reglas y *outputs* es el número de salidas del modelo.
        
        Returns:
            torch.tensor: Tensor de tamaño (outputs, batch_size, rules) que contiene las salidas individuales de cada regla sin ponderar con los normalized firing levels.
        
        """
        with torch.no_grad():
            outputs = torch.bmm(x.unsqueeze(0).expand(consequents[:, :, :-1].size(0), -1, -1), torch.transpose(consequents[:, :, :-1], 1, 2)) + consequents[:, :, -1].unsqueeze(1)
        return outputs

    def random_consequents(self, outputs, rules, input_size, dtype):
        """
        Inicializa los parámetros consecuentes de la red ANFIS de manera aleatoria en el rango [-1, 1].
        """
        return 2 * torch.rand(outputs, rules, input_size + 1, dtype=dtype) - 1