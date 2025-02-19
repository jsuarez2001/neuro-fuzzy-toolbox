import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF, Linear_CF
from neuro_fuzzy_toolbox.layers import (
    h_FuzzificationLayer, 
    h_FiringLevelsLayer, 
    FuzzificationLayer,
    FiringLevelsLayer,
    NormalizationLayer, 
    ConsequentLayer, 
    OutputLayer
    )


class base_ANFIS(nn.Module):
    """
    Clase base para un sistema de inferencia neuro-difuso adaptativo (ANFIS). Esta clase contiene los métodos y atributos comunes en los modelos ANFIS implementados en el toolbox. 
    Las clases ANFIS y h_ANFIS heredan de esta clase y añaden funcionalidades específicas.
    
    Esta clase no debe ser instanciada directamente.
    """
    def __init__(self):
        super(base_ANFIS, self).__init__()
    
    # ---- Forward pass ----
    def forward(self, x, return_probabilities=False):
        """
        Realiza un paso hacia adelante a través del modelo.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            return_probabilities (bool): Indica si el resultado pasará por una función Softmax para obtener probabilidades. Solo se aplica si el tipo de salida es 'multiclass', en caso contrario, se ignora (Default: False).
        
        """
        output = self._fuzzification_layer(x)
        output = self._consequent_layer(x, self._normalization_layer(self._firing_levels_layer(output)))
        output = self._output_layer(output, return_probabilities)
        return output
    
    
    # ---- Initialize parameters ----
    def init_premises(self, x):
        """
        Inicializa los parámetros de las funciones de membresía de la capa de fuzzificación del modelo a partir de los datos ingresados.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
        """
        self._dtype = x.dtype
        self._consequent_layer._to_dtype(x.dtype) # Set dtype to consequents
        self._fuzzification_layer.init_premises(x)
        self._fuzzification_layer._membership_function._max_val_plot = x.max().item()
        self._fuzzification_layer._membership_function._min_val_plot = x.min().item()
        
    
    def init_consequents(self, x, y):
        """
        Inicializa los parámetros consecuentes del modelo usando una estimación de mínimos cuadrados.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            y (torch.Tensor): Tensor con los datos de salida. Es de tamaño (batch_size, outputs).
        """
        _, w_norm, _ = self.intermediate_values(x)
        xe = torch.cat([x, torch.ones(x.shape[0], 1, dtype=self._dtype)], dim=1)
        fs = w_norm.unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
        X = xe.repeat(1, self.rules)
        
        '''preliminary fix for the dtype issue'''
        if self._output_type == 'multiclass':
            y = torch.nn.functional.one_hot(y, self._outputs)
        if y.dtype != X.dtype:
            y = y.to(X.dtype)
        '''preliminary fix for the dtype issue'''
        
        # Solve least squares problem using QR decomposition with pivoting
        C, _, _, _ = torch.linalg.lstsq(X * fs, y)
        new_consequents = C.t().reshape(self._outputs, self.rules, xe.shape[1])
        
        # Update consequents
        self.set_consequents(new_consequents)
        
    
    # ---- Model predict ----
    def predict(self, x):
        """
        Realiza una predicción con el modelo. Ajusta la salida a la forma esperada según el tipo de salida del modelo.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            
        Returns:
            np.ndarray: Predicciones del modelo.
        """
        with torch.no_grad():
            output = self.forward(x).detach().numpy()
        
        if self._output_type == 'multiclass':
            output = self.forward(x, return_probabilities=True).detach().numpy()
            output = np.argmax(output, axis=1, keepdims=True)
            output = np.squeeze(np.transpose(output))
            
        elif self._output_type == 'binary':
            output = (output > 0.5).astype(int)
            
        return output
    
    
    # ---- Intermediate values ----
    def intermediate_values(self, x):
        """
        Emula un paso hacia adelante del modelo y retorna los valores intermedios de las capas del modelo.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            
        Returns:
            tuple: Tupla con los valores intermedios de las capas del modelo. Contiene:
                - w: Niveles de disparo.
                - w_norm: Niveles de disparo normalizados.
                - outputs: Salidas individuales de las reglas del modelo.
        """
        with torch.no_grad():
            w = self._fuzzification_layer(x)
            w = self._firing_levels_layer(w)
            w_norm = self._normalization_layer(w)
            outputs = self._consequent_layer(x, w_norm)
        return w, w_norm, outputs
    
    
    # ---- ANFIS parameters info ----
    @property
    def rules(self):
        """
        Retorna la cantidad de reglas del modelo ANFIS.
        
        Returns:
            int: Cantidad de reglas.
        """
        return self.get_consequents().shape[1]
    
    
    # ----- Consequents seters and getters -----
    def set_consequents(self, consequents):
        """
        Setea los parámetros consecuentes del modelo.
        
        Args:
            consequents (torch.tensor): Tensor con los parámetros consecuentes. Su forma debe ser (outputs, rules, input_size + 1).
        """
        self._consequent_layer._consequents = Parameter(consequents, requires_grad=True)
    
    def get_consequents(self):
        """
        Retorna los parámetros consecuentes del modelo.
        
        Returns:
            torch.tensor: Tensor con los parámetros consecuentes. Su forma es (outputs, rules, input_size + 1).
        """
        return self._consequent_layer._consequents.data.clone().detach()
    
    def get_consequents_as_parameters_list(self):
        """
        Retorna los parámetros consecuentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            list: Lista de 1 solo elemento (nn.Parameter) que contiene los parámetros consecuentes.
        """
        return [self._consequent_layer._consequents]
    
    
    # ----- Parameters dataframes -----
    @property
    def premises_structure(self):
        """
        Retorna la estructura de los parámetros de las funciones de membresía.
        
        Returns:
            pandas.DataFrame: DataFrame con la estructura de los parámetros de las funciones de membresía.
        """
        return self._fuzzification_layer.premises_structure
    
    @property
    def consequents_structure(self):
        """
        Retorna la estructura de los parámetros consecuentes.
        
        Returns:
            list: Lista de DataFrames de pandas que contienen la estructura de los parámetros consecuentes.
        """
        return self._consequent_layer.consequents_structure
    
    def show_premises_structure(self):
        """
        Impresión de la estructura de los parámetros de las funciones de membresía.
        
        """
        print(self.premises_structure)
        
    def show_consequents_structure(self):
        """
        Impresión de la estructura de los parámetros consecuentes.
        """
        output = 1
        for df in self.consequents_structure:
            print(f'- Output {output}:')
            print(df)
            print('\n')
            output += 1
    
    
    # ----- Plot premises -----
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False):
        """
        Plotea las funciones de membresía de los antecedentes del modelo.
        
        Args:
            mf (int): Índice de la función de membresía a plotear. Si es None, se plotean todas las funciones de membresía.
            input_dim (int): Dimensión de la entrada a plotear. Si es None, se plotean todas las dimensiones.
            group_by_dim (bool): Si es True, agrupa las funciones de membresía en un solo gráfico por cada dimensión de entrada. (Default: False)
        """
        self._fuzzification_layer.plot_premises(mf, input_dim, group_by_dim)
    


class h_ANFIS(base_ANFIS):
    """
    Clase para un sistema de inferencia neuro-difuso adaptativo (ANFIS) homogéneo, es decir, con la misma cantidad de funciones de membresía para cada feature de los datos de entrada, 
    limitando a que cada uno con el mismo número de variables lingüisticas.
    
    Esta clase tiene un parámetro especial 'rule_reduced' que permite reducir en número de reglas generadas en el cálculo de los niveles de disparo. Esto se logra evitando hacer la combinatoria completa para las multiplicaciones de los valores de pertenencia. 
    El procedimiento entonces es una multiplicación solo entre los valores de pertenencia *i* de cada feature, dando como resultado una cantidad de reglas igual al número de funciones de membresía de cada feature.
    """
    
    def __init__(self, input_size, num_mfs, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, output_type="regression", rule_reduced=False, dtype=torch.float32):
        super(h_ANFIS, self).__init__()
        """
        Inicializa un modelo ANFIS homogéneo.
        
        Args:
            input_size (int): Número de features de los datos de entrada.
            num_mfs (int): Número de funciones de membresía por feature.
            outputs (int): Número de salidas del modelo (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            consequent_function (ConsequentFunction): Función consecuente a utilizar (Default: Linear_CF).
            output_type (str): Tipo de salida del modelo (Default: 'regression').
            rule_reduced (bool): True si se desea utilizar la versión reducida de reglas, False en caso contrario (Default: False).
            dtype (torch.dtype): Tipo de dato a utilizar en el modelo (Default: torch.float32).
        """
        
        if rule_reduced:
            rules = num_mfs
        else:
            rules = num_mfs**input_size
        
        
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        
        
        # ANFIS structure info
        self._rule_reduced = rule_reduced
        
        
        # Output info
        self._output_type = output_type
        self._outputs = outputs
        
        
        # Layers
        self._fuzzification_layer = h_FuzzificationLayer(
            input_size=input_size,
            num_mfs=num_mfs, 
            membership_function=membership_function, 
            dtype=dtype
            )
        
        self._firing_levels_layer = h_FiringLevelsLayer(rule_reduced=rule_reduced)
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = ConsequentLayer(
            input_size=input_size,
            rules=rules,
            outputs=outputs,
            consequent_function=consequent_function, 
            dtype=dtype
            )
        
        self._output_layer = OutputLayer(output_type=self._output_type)
        
    
    # ----- Premises seters and getters -----
    def set_premises(self, premises):
        """
        Setea los parámetros de las funciones de membresía de la capa de fuzzificación del modelo.
        
        Args:
            premises (torch.tensor): Tensor con los parámetros de las funciones de membresía. Su forma debe ser (input_size, num_mfs, mf_params), donde mf_params es el número de parámetros de la función de membresía.
        """
        self._fuzzification_layer._premises = Parameter(premises, requires_grad=True)
        
    def get_premises(self):
        """
        Retorna los antecedentes del modelo.
        
        Returns:
            torch.tensor: Tensor con los antecedentes del modelo. Su forma es (input_size, num_mfs, mf_params).
        """
        return self._fuzzification_layer._premises.data.clone().detach()
    
    def get_premises_as_parameters_list(self):
        """
        Retorna los antecedentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            list: Lista de 1 solo elemento (nn.Parameter) que contiene los parámetros de los antecedentes.
        """
        return [self._fuzzification_layer._premises]
    
    
    # ---- ANFIS parameters info ----
    @property
    def num_mfs(self):
        """
        Retorna la cantidad de funciones de membresía por feature.
        
        Returns:
            int: Cantidad de funciones de membresía que tiene cada feature.
        """
        return self.get_premises().shape[1]
    
    # ----- Load state dict -----
    def load_state_dict(self, state_dict):
        """
        Carga un estado del modelo.
        
        Args:
            state_dict (dict): Diccionario con el estado del modelo.
        """
        self.set_premises(state_dict['_fuzzification_layer._premises'])
        self.set_consequents(state_dict['_consequent_layer._consequents'])
        


class ANFIS(base_ANFIS):
    """
    Clase para un sistema de inferencia neuro-difuso adaptativo (ANFIS) con una cantidad de funciones de membresía arbitraria para cada feature de los datos de entrada.
    """
    
    def __init__(self, mf_distribution, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, output_type="regression", dtype=torch.float32):
        """
        Inicializa un modelo ANFIS.
        
        Args:
            mf_distribution (list): Distribución de funciones de membresía por feature. Debe ser una lista con la cantidad de funciones de membresía por feature.
            outputs (int): Número de salidas del modelo (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            consequent_function (ConsequentFunction): Función consecuente a utilizar (Default: Linear_CF).
            output_type (str): Tipo de salida del modelo (Default: 'regression').
            dtype (torch.dtype): Tipo de dato a utilizar en el modelo (Default: torch.float32).
            
        """
        super(ANFIS, self).__init__()
        # Input data info
        self._input_size = len(mf_distribution)
        self._dtype = dtype
        
        # ANFIS structure info
        self._mf_distribution = torch.tensor(mf_distribution)
        self._rules = self._mf_distribution.prod().item()
        self._mfs = self._mf_distribution.sum().item()
        
        # Output info
        self._output_type = output_type
        self._outputs = outputs
        
        # Layers
        self._fuzzification_layer = FuzzificationLayer(
            mf_distribution=mf_distribution, 
            membership_function=membership_function,
            dtype=dtype)
        
        self._firing_levels_layer = FiringLevelsLayer(
            mf_distribution=self._mf_distribution
            )
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = ConsequentLayer(
            input_size=self._input_size,
            rules=self._rules,
            outputs=outputs,
            consequent_function=consequent_function,
            dtype=dtype)
        
        self._output_layer = OutputLayer(output_type=output_type)
    
    
    # ----- Premises seters and getters -----
    def get_premises(self):
        """
        Retorna los antecedentes del modelo.
        
        Returns:
            list: Lista de tensores con los antecedentes del modelo asociados a cada feature (por lo que el largo de la lista es input_size). Cada tensor tiene forma (num_mfs, mf_params), donde num_mfs es la cantidad de funciones de membresía para el feature asociado en cuestión y mf_params es el número de parámetros de la función usada.
        """
        return [mf.data.clone().detach() for mf in self._fuzzification_layer._premises]
    
    def set_premises(self, premises):
        """
        Setea los parámetros de las funciones de membresía de la capa de fuzzificación del modelo.
        
        Args:
            premises (list): Lista de tensores con los parámetros de las funciones de membresía. Cada tensor debe tener forma (num_mfs, mf_params), donde mf_params es el número de parámetros de la función de membresía.
        """
        for i, premise in enumerate(premises):
            self._fuzzification_layer._premises[i] = Parameter(premise, requires_grad=True)
            
    def get_premises_as_parameters_list(self):
        """
        Retorna los antecedentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            nn.ParameterList: Lista de parámetros que contiene los antecedentes.
        """
        return self._fuzzification_layer._premises


    # ---- ANFIS parameters info ----
    @property
    def num_mfs(self):
        """
        Retorna la cantidad de funciones de membresía por feature.
        
        Returns:
            torch.tensor: Tensor con la cantidad de funciones de membresía por feature.
        """
        return self._mf_distribution
    
    # ----- Load state dict -----
    def load_state_dict(self, state_dict):
        """
        Carga un estado del modelo.
        
        Args:
            state_dict (dict): Diccionario con el estado del modelo.
        """
        prems = []
        for i in range(self._input_size):
            prems.append(state_dict['_fuzzification_layer._premises.' + str(i)])
        self.set_premises(prems)
        self.set_consequents(state_dict['_consequent_layer._consequents'])