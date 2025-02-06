import torch
import torch.nn as nn
from torch.nn import Parameter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF
from neuro_fuzzy_toolbox.layers import (
    h_FuzzificationLayer, 
    h_FiringLevelsLayer, 
    FuzzificationLayer,
    FiringLevelsLayer,
    NormalizationLayer, 
    )

class h_AntecedentBlock(nn.Module):
    """
    Bloque de antecedentes de un modelo ANFIS homogéneo (todos los features tienen las misma cantidad de funciones de membresía). Este bloque se encarga de realizar la fuzzificación de los datos de entrada y calcular los niveles de disparo de las reglas y normalizarlos si se desea.
    
    """
    def __init__(self, input_size, num_mfs, normalization=True, membership_function=GeneralizedBell_MF, dtype=torch.float32):
        super(h_AntecedentBlock, self).__init__()
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        
        # Membership functions info
        self._membership_function = membership_function()
        
        # Normalization flag
        self._normalization = normalization

        # Layers
        self._fuzzification_layer = h_FuzzificationLayer(
            input_size=input_size, 
            num_mfs=num_mfs, 
            membership_function=membership_function, 
            dtype=dtype
            )
        self._firing_levels_layer = h_FiringLevelsLayer()
        
        if normalization:
            self._normalization_layer = NormalizationLayer()
        else:
            self._normalization_layer = nn.Identity()
            
        # Firing levels generated
        self._n_firing_levels_generated = num_mfs**input_size
        
        
    @property
    def n_firing_levels_generated(self):
        return self._n_firing_levels_generated
            
            
    def forward(self, x):
        """
        Paso hacia adelante del bloque de antecedentes.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            
        Returns:
            torch.Tensor: Tensor con los niveles de disparo normalizados. Es de tamaño (batch_size, num_mfs**input_size).
        """
        output = self._fuzzification_layer(x)
        output = self._firing_levels_layer(output)
        return self._normalization_layer(output)
    
    
    def init_premises(self, x):
        """
        Inicializa los parámetros de las funciones de membresía de la capa de fuzzificación del modelo a partir de los datos ingresados.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
        """
        self._dtype = x.dtype
        self._fuzzification_layer.init_premises(x)
        
        
    # ---- Intermediate values ----
    def intermediate_values(self, x):
        """
        Emula un paso hacia adelante del bloque y retorna los valores intermedios del mismo.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            
        Returns:
            tuple: Tupla con los valores intermedios de las capas del modelo. Contiene:
                - w: Niveles de disparo.
                - w_norm: Niveles de disparo normalizados. (Si existe capa de normalización).
        """
        with torch.no_grad():
            w = self._fuzzification_layer(x)
            w = self._firing_levels_layer(w)
            if self.normalization:
                w_norm = self._normalization_layer(w)
            w_norm = None
        return w, w_norm
        
    
    # ----- Parameters dataframes -----
    @property
    def premises_structure(self):
        """
        Retorna la estructura de los parámetros de las funciones de membresía.
        
        Returns:
            pandas.DataFrame: DataFrame con la estructura de los parámetros de las funciones de membresía.
        """
        return self._fuzzification_layer.premises_structure
    
    def show_premises_structure(self):
        """
        Impresión de la estructura de los parámetros de las funciones de membresía.
        
        """
        print(self.premises_structure)
        
    
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
    
    
class AntecedentBlock(nn.Module):
    def __init__(self, mf_distribution, normalization=True, membership_function=GeneralizedBell_MF, dtype=torch.float32):
        super(AntecedentBlock, self).__init__()
        # Input data info
        self._input_size = len(mf_distribution)
        self._dtype = dtype
        
        # Membership functions info
        self._mf_distribution = torch.tensor(mf_distribution)
        self._membership_function = membership_function()
        self._mfs = self._mf_distribution.sum().item()
        
        # Normalization flag
        self._normalization = normalization

        # Layers
        self._fuzzification_layer = FuzzificationLayer(
            mf_distribution=mf_distribution, 
            membership_function=membership_function,
            dtype=dtype
            )
        self._firing_levels_layer = FiringLevelsLayer(
            mf_distribution=self._mf_distribution
            )
        
        if normalization:
            nl = NormalizationLayer()
        else:
            nl = nn.Identity()
        self._normalization_layer = nl
        
        # Firing levels generated
        self._n_firing_levels_generated = self._mf_distribution.prod().item()
        
        
    @property
    def n_firing_levels_generated(self):
        return self._n_firing_levels_generated
            
            
    def forward(self, x):
        output = self._fuzzification_layer(x)
        output = self._firing_levels_layer(output)
        return self._normalization_layer(output)
    
    
    def init_premises(self, x):
        """
        Inicializa los parámetros de las funciones de membresía de la capa de fuzzificación del modelo a partir de los datos ingresados.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
        """
        self._dtype = x.dtype
        self._fuzzification_layer.init_premises(x)
        
        
    # ---- Intermediate values ----
    def intermediate_values(self, x):
        """
        Emula un paso hacia adelante del bloque y retorna los valores intermedios del mismo.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            
        Returns:
            tuple: Tupla con los valores intermedios de las capas del modelo. Contiene:
                - w: Niveles de disparo.
                - w_norm: Niveles de disparo normalizados. (Si existe capa de normalización).
        """
        with torch.no_grad():
            w = self._fuzzification_layer(x)
            w = self._firing_levels_layer(w)
            if self.normalization:
                w_norm = self._normalization_layer(w)
            w_norm = None
        return w, w_norm
        

    # ----- Parameters dataframes -----
    @property
    def premises_structure(self):
        """
        Retorna la estructura de los parámetros de las funciones de membresía.
        
        Returns:
            pandas.DataFrame: DataFrame con la estructura de los parámetros de las funciones de membresía.
        """
        return self._fuzzification_layer.premises_structure
    
    def show_premises_structure(self):
        """
        Impresión de la estructura de los parámetros de las funciones de membresía.
        
        """
        print(self.premises_structure)
        
        
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