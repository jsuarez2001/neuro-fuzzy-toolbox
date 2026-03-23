import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    """
    Clase para representar la capa de salida de un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS).
    Esta capa se encarga de calcular la salida final del modelo ANFIS a partir las salidas de cada regla.
    
    Tiene naturaleza dependiente del tipo de salida del modelo ANFIS, por lo que se debe especificar el tipo de salida al inicializar la capa. Los tipos de salida soportados son:
        - 'default': Para problemas de regresión.
        - 'sigmoid': Para problemas de clasificación binaria.
        - 'softmax': Para problemas de clasificación multiclase.
        
    """
    def __init__(self, output_type):
        """
        Inicializa una nueva instancia de la clase OutputLayer.
        
        Args:
            output_type (str): Tipo de salida del modelo ANFIS. Puede ser 'default', 'sigmoid' o 'softmax'.
        """
        super(OutputLayer, self).__init__()
        self._output_type = output_type.lower()
        
        if (self._output_type == 'default' or self._output_type == 'softmax'):
            self._last_layer = nn.Identity()
        elif (self._output_type == 'sigmoid'):
            self._last_layer = nn.Sigmoid()
            
        if self._output_type != 'softmax':
            self._get_output = lambda rules_outputs, return_probs: torch.sum(rules_outputs, dim=-1).t().squeeze(1)
        else:
            self._get_output = lambda rules_outputs, return_probs: nn.functional.softmax(torch.sum(rules_outputs, dim=-1).t().squeeze(1), dim=1) if return_probs else torch.sum(rules_outputs, dim=-1).t().squeeze(1)
        

    def forward(self, rules_outputs, return_probs=False):
        """
        Realiza un paso hacia adelante para calcular la salida de la capa de salida.
        
        Args:
            rules_outputs (torch.tensor): Tensor de tamaño (batch_size, rules) que contiene las salidas de cada regla.
            return_probs (bool): Indica si el resultado pasará por una función Softmax para obtener probabilidades. Solo se aplica si el tipo de salida es 'softmax', en caso contrario, se ignora (Default: False).
        """
        return self._last_layer(self._get_output(rules_outputs, return_probs))