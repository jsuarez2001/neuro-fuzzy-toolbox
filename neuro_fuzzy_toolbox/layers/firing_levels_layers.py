import torch
import torch.nn as nn


class FiringLevelsLayer(nn.Module):
    """
    Clase para representar la capa que calcula los niveles de disparo en un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS). Esta capa emula las operaciones AND entre los valores de pertenencia de las reglas.
    
    Esta diseñada para un modelo ANFIS general, es decir, para manejar diferentes cantidades de funciones de membresía para cada feature de los datos de entrada.
    """
    def __init__(self, mf_distribution):
        """
        Inicializa una nueva instancia de la clase FiringLevelsLayer.
        
        Args:
            mf_distribution (list): Lista de listas que contiene la distribución de las funciones de membresía para cada feature de los datos de entrada.
        """
        super(FiringLevelsLayer, self).__init__()
        self._firing_level_mask = (torch.arange(mf_distribution.max()).unsqueeze(1) < mf_distribution).t()
        self._rules = mf_distribution.prod()

    def forward(self, membership_values):
        """
        Realiza un paso hacia adelante a través de la capa para calcular los niveles de disparo.
        
        Args:
            membership_values (torch.Tensor): Tensor de entrada que contiene los valores de pertenencia para cada regla. Debe tener tamaño (batch_size, input_size, max_num_mfs), donde max_n_mfs es el número máximo de funciones de membresía entre todos los features.
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, num_mfs1 * num_mfs2 * ... * num_mfsn) que contiene los niveles de disparo.
        """
        return torch.cat([torch.cartesian_prod(*[dim_mvs[dim_mask] for dim_mvs, dim_mask in zip(mvs, self._firing_level_mask)]).prod(dim=-1) for mvs in membership_values]).reshape(-1, self._rules)



class h_FiringLevelsLayer(nn.Module):
    """
    Clase para representar la capa que calcula los niveles de disparo en un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS) homogéneo, es decir, 
    con la misma cantidad de funciones de membresía para cada feature de los datos de entrada. Esta capa emula las operaciones AND entre los valores de pertenencia de las reglas.
    
    Cuenta con un parámetro especial 'rule_reduced' que permite reducir en número de reglas generadas en el cálculo de los niveles de disparo. Esto se logra evitando hacer la combinatoria completa para las multiplicaciones de los valores de pertenencia.
    El procedimiento en este caso se realizaría solo multiplicando entre sí los valores de pertenencia *i* de cada feature, dando como resultado una cantidad de reglas igual al número de funciones de membresía de cada feature. (esto se detalla de mejor manera en :ref:`rule-reduced ANFIS <rule-reduced ANFIS>`).
    """
    def __init__(self, rule_reduced=False):
        """
        Inicializa una nueva instancia de la clase h_FiringLevelsLayer.
        
        Args:
            rule_reduced (bool): Indica si se deben reducir las reglas para realizar el cálculo de los niveles de disparo (Default: False).
        """
        super(h_FiringLevelsLayer, self).__init__()
        if rule_reduced:
            self._get_firing_levels = lambda membership_values: membership_values.prod(dim=membership_values.dim()-2)
        else:
            self._get_firing_levels = lambda membership_values: torch.cat([torch.cartesian_prod(*torch.unbind(t, dim=0)).prod(dim=-1) for t in membership_values]).reshape(-1, membership_values.shape[-1]**membership_values.shape[-2])

    def forward(self, membership_values):
        """
        Realiza un paso hacia adelante a través de la capa para calcular los niveles de disparo.
        
        Args:
            membership_values (torch.Tensor): Tensor de entrada que contiene los valores de pertenencia para cada regla. Debe tener tamaño (batch_size, input_size, num_mfs), donde num_mfs es el número de funciones de membresía para cada feature.
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, num_mfs^input_size) que contiene los niveles de disparo. Si es 'rule_reduced' es True, el tamaño será (batch_size, num_mfs).
            
        """
        return self._get_firing_levels(membership_values)
    
    
    
class rule_reduced_FiringLevelsLayer(nn.Module):
    """
    Clase para representar la capa que calcula los niveles de disparo en un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS) homogéneo de reglas reducidas, es decir, 
    con la misma cantidad de funciones de membresía para cada feature de los datos de entrada, pero con reducción de reglas. Esta capa emula las operaciones AND entre los valores de pertenencia de las reglas.
    
    La reducción de reglas se logra evitando hacer la combinatoria completa para las multiplicaciones de los valores de pertenencia.
    El procedimiento en este caso se realizaría solo multiplicando entre sí los valores de pertenencia *i* de cada feature, dando como resultado una cantidad de reglas igual al número de funciones de membresía de cada feature. (esto se detalla de mejor manera en :ref:`rule-reduced ANFIS <rule-reduced ANFIS>`).
    
    
    Cuenta con un parámetro especial 'default_rule' que permite agregar un nivel de disparo extra para capturar todas las combinaciones que el conjunto de reglas de los modelos con reglas reducidas no capturan. (esto se detalla de mejor manera en :ref:`PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA PONER ALGO ACA <rule-reduced ANFIS>`).
    """
    def __init__(self, default_rule=False):
        """
        Inicializa una nueva instancia de la clase rule_reduced_FiringLevelsLayer.
        
        Args:
            rule_reduced (bool): Indica si se deben reducir las reglas para realizar el cálculo de los niveles de disparo (Default: False).
        """
        super(rule_reduced_FiringLevelsLayer, self).__init__()
        if default_rule:
            self._get_firing_levels = lambda firing_levels, input_size: torch.cat((firing_levels, torch.pow(1 - firing_levels.max(dim=1).values.unsqueeze(1), input_size)), dim=1)
        else:
            self._get_firing_levels = lambda firing_levels, input_size: firing_levels
            
    def forward(self, membership_values):
        """
        Realiza un paso hacia adelante a través de la capa para calcular los niveles de disparo.
        
        Args:
            membership_values (torch.Tensor): Tensor de entrada que contiene los valores de pertenencia para cada regla. Debe tener tamaño (batch_size, input_size, num_mfs), donde num_mfs es el número de funciones de membresía para cada feature.
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, num_mfs^input_size) que contiene los niveles de disparo. Si es 'rule_reduced' es True, el tamaño será (batch_size, num_mfs).
            
        """
        firing_levels = membership_values.prod(dim=membership_values.dim()-2)
        return self._get_firing_levels(firing_levels, membership_values.shape[1])



class NormalizationLayer(nn.Module):
    """
    Clase para representar la capa que normaliza los niveles de disparo en un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS).
    
    Cuenta con un parámetro especial 'default_rule' que permite agregar un nivel de disparo extra para capturar todas las combinaciones que el conjunto de reglas de los modelos con reglas 
    reducidas no capturan.
    """
    def __init__(self, default_rule=False):
        """
        Inicializa una nueva instancia de la clase h_FiringLevelsLayer.
        
        Args:
            default_rule (bool): Indica si se debe agregar un nivel de disparo adicional para simular una regla por defecto (Default: False).
        """
        super(NormalizationLayer, self).__init__()
        if default_rule:
            self._get_firing_levels = lambda firing_levels: firing_levels[:,:-1]
        else:
            self._get_firing_levels = lambda firing_levels: firing_levels

    def forward(self, w):
        """
        Realiza un paso hacia adelante a través de la capa para normalizar los niveles de disparo.
        
        Args:
            w (torch.Tensor): Tensor de entrada que contiene los niveles de disparo. Debe tener tamaño (batch_size, num_rules). donde num_rules es el número de reglas generadas en el modelo ANFIS, el cual dependerá del modelo específico.
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, num_rules) que contiene los niveles de disparo normalizados.
        """
        total = torch.sum(w, dim=-1, keepdim=True)
        total[total == 0] += 1
        norm_firing_levels = self._get_firing_levels(w)/total
        return norm_firing_levels
    
    def get_default_rule_firing_level(self, firing_levels):
        """
        Realiza un paso hacia adelante a través de la capa para normalizar los niveles de disparo.
        
        Args:
            firing_levels (torch.Tensor): Tensor de entrada que contiene los niveles de disparo. Debe tener tamaño (batch_size, num_rules). donde num_rules es el número de reglas generadas en el modelo ANFIS, el cual dependerá del modelo específico.
            
        Returns:
            tuple: Tupla con los niveles de disparo de la regla por defecto en tensores de un elemento. Contiene:
                - default_w (torch.Tensor): Nivel de disparo de la regla por defecto.
                - default_w_norm (torch.Tensor): Nivel de disparo de la regla por defecto normalizado.
        """
        firing_levels = self._get_firing_levels(firing_levels)