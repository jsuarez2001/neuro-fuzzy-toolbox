import torch
import torch.nn as nn
from torch.nn import Parameter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF

class FuzzificationLayer(nn.Module):
    """
    Clase para representar la capa de fuzzificación de un sistema de inferencia neuro-difuso adaptativo (ANFIS). Esta capa se encarga de transformar los datos de entrada en valores de membresía para cada función de membresía de cada feature.

    Esta diseñada para un modelo ANFIS general, es decir, para manejar diferentes cantidades de funciones de membresía para cada feature de los datos de entrada.
    """
    def __init__(self, mf_distribution, membership_function=GeneralizedBell_MF, dtype=torch.float32):
        """
        Inicializa una nueva instancia de FuzzificationLayer.
        
        Args:
            mf_distribution (list): Lista de enteros que representan la cantidad de funciones de membresía para cada feature de entrada.
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            dtype (torch.dtype): Tipo de dato de los datos de entrada (Default: torch.float32).
        
        """
        super(FuzzificationLayer, self).__init__()
        
        # Input data info
        self._input_size = len(mf_distribution)
        self._dtype = dtype
        
        # Membership function
        self._membership_function = membership_function()
        
        # Premise parameters
        self._max_n_mfs = max(mf_distribution)
        self._mf_distribution = mf_distribution
        self._premises = self._random_premises()
        
    
    def forward(self, x):
        """
        Realiza un paso hacia adelante para calcular los valores de membresía para cada feature de entrada.
        
        Args:
            x (torch.Tensor): Conjunto de datos de entrada, tiene tamaño (batch_size, input_size).
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, input_size, max_n_mfs) que contiene los valores de membresía para cada feature, donde max_n_mfs es el número máximo de funciones de membresía entre todos los features.
            
        Note:
            - Para manejar las distintas cantidades de funciones de membresía para cada feature, se rellenan con ceros las funciones de membresía faltantes. Esto podría ser mejorable en futuras versiones (agregando 0s directamente a un tensor que contenga los parámetros directamente).
        """
        reshaped_premises = torch.tensor([], dtype=self._dtype)
                
        # Reshape premises
        for mf in self._premises:
            if mf.size(0) < self._max_n_mfs:
                mf = torch.cat((mf, torch.zeros(self._max_n_mfs - mf.size(0), mf.size(1), dtype=self._dtype)), 0)
            reshaped_premises = torch.cat((reshaped_premises, mf.unsqueeze(dim=0)), 0)
            
        return self._membership_function(x, reshaped_premises)
    
    
    def init_premises(self, x_train):
        """
        Inicializa los parámetros de las funciones de membresía de los antecedentes utilizando los datos de entrenamiento.
        
        Args:
            x_train (torch.Tensor): Datos de entrenamiento de entrada.
        """
        self._dtype = x_train.dtype
        self._premises = nn.ParameterList(self._membership_function.general_initialize_premises(x_train, self._mf_distribution))
    
    
    def _random_premises(self):
        """
        Genera parámetros de funciones de membresía aleatorios para los antecedentes.
        """
        premises = nn.ParameterList()
        for n_mfs in self._mf_distribution:
           premises.append(Parameter(self._membership_function.random_single_feature_mfs(n_mfs, self._dtype)))
        return premises
    
    
    @property
    def premises_structure(self):
        """
        Retorna la estructura de los antecedentes.
        
        Returns:
            pd.DataFrame: DataFrame de pandas que contiene la estructura de los antecedentes.
        """
        df = pd.DataFrame()
        mfs = ['MF {}'.format(i) for i in range(0, self._max_n_mfs)]
        
        for i in range(self._input_size):
            num_mfs = self._mf_distribution[i]
            for j, param_name in enumerate(self._membership_function._params):
                column_data = [self._premises[i][mf_idx, j].item() if mf_idx < num_mfs else None for mf_idx in range(self._max_n_mfs)]
                column = pd.Series(column_data, index=mfs, name=param_name + f' (x{i})')
                df[param_name + f' (x{i})'] = column

        return df
    
    
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False):
        """
        Plotea las funciones de membresía de los antecedentes.
        
        Args:
            mf (int): Función de membresía a plotear. Si es None, se plotean todas las funciones de membresía (Default: None).
            input_dim (int): Dimensión de entrada a plotear. Si es None, se plotean todas las dimensiones de entrada (Default: None).
            group_by_dim (bool): Si es True, agrupa las funciones de membresía en un solo gráfico por cada dimensión de entrada. (Default: False)
        """
        variables = [f'x{i}' for i in range(self._input_size)]
        dataframe = self.premises_structure
        
        x = np.linspace(self._membership_function._max_val_plot, self._membership_function._min_val_plot, 500)
        
        # Determine which mfs and dimensions to plot
        if mf is not None:
            mf = f'MF {mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"MF '{mf}' not found in premises. Available mfs: {dataframe.index.tolist()}")
            mfs_to_plot = [mf]
        else:
            mfs_to_plot = dataframe.index
            
        # Validate input dimension
        if input_dim is not None:
            if not isinstance(input_dim, int) or input_dim < 0 or input_dim >= len(variables):
                raise ValueError(f"input_dim must be between 0 and {len(variables)-1}")
            dims_to_plot = [input_dim]
        else:
            dims_to_plot = range(len(variables))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(6, 4*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf in mfs_to_plot:
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, f'{param} ({var})']
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax.plot(x, y, label=f'{mf}')
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}')
                ax.grid(True)
                ax.set_xlabel('x')
                ax.set_ylabel('Membership Value')
                ax.legend()

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)

            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_mfs, ncols=n_dims, figsize=(5*n_dims, 4*n_mfs), squeeze=False)

            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, f'{param} ({var})']
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax = axes[i, j]
                            ax.plot(x, y, label=f'{mf}, {var}')
                            ax.set_title(f'{mf}, {var}')
                            ax.grid(True)
                            if i == n_mfs - 1:
                                ax.set_xlabel('x')
                            if j == 0:
                                ax.set_ylabel('Membership Value')
                    except KeyError as e:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
            plt.tight_layout()
            plt.show()



class h_FuzzificationLayer(nn.Module):
    """
    Clase para representar la capa de fuzzificación de un sistema de inferencia neuro-difuso adaptativo (ANFIS) homogéneo, es decir, 
    con la misma cantidad de funciones de membresía para cada feature de los datos de entrada, limitando a que cada uno tenga el mismo 
    número de variables lingüisticas.
    """
    def __init__(self, input_size, num_mfs=1, membership_function=GeneralizedBell_MF, dtype=torch.float32):
        """
        Inicializa una nueva instancia de h_FuzzificationLayer.
        
        Args:
            input_size (int): Número de features de entrada.
            num_mfs (int): Número de funciones de membresía para cada feature (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            dtype (torch.dtype): Tipo de dato de los datos de entrada (Default: torch.float32).
        """
        super(h_FuzzificationLayer, self).__init__()

        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        
        # Membership function
        self._membership_function = membership_function()

        # Initialize premise parameters
        self._premises = Parameter(membership_function().random_premises(input_size, num_mfs, dtype), requires_grad=True)



    def forward(self, x):
        """
        Realiza un paso hacia adelante para calcular los valores de membresía para cada feature de entrada.
        
        Args:
            x (torch.Tensor): Conjunto de datos de entrada, tiene tamaño (batch_size, input_size).
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, input_size, num_mfs) que contiene los valores de membresía para cada feature, donde num_mfs es el número de funciones de membresía para cada feature.
        """
        return self._membership_function(x, self._premises)
    
    
    
    def init_premises(self, x_train):
        """
        Inicializa los parámetros de las funciones de membresía de los antecedentes utilizando los datos de entrenamiento.
        
        Args:
            x_train (torch.Tensor): Datos de entrenamiento de entrada.
        """
        self._dtype = x_train.dtype
        self._premises = Parameter(self._membership_function.initialize_premises(x_train=x_train, num_mfs=self._premises.data.shape[1]), requires_grad=True)



    @property
    def premises_structure(self):
        """
        Retorna la estructura de los antecedentes en un DataFrame de pandas.
        """
        df = pd.DataFrame()
        mfs = ['MF {}'.format(i) for i in range(0, self._premises.data.shape[1])]
        
        for i in range(self._input_size):
            for j in range(len(self._membership_function._params)):
                column = pd.Series(self._premises.data[i,:,j], index=mfs, name=self._membership_function._params[j] + f' (x{i})', )
                df[self._membership_function._params[j] + f' (x{i})'] = column

        return df
    
    
    
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False):
        """
        Plotea las funciones de membresía de los antecedentes.
        
        Args:
            mf (int): Función de membresía a plotear. Si es None, se plotean todas las funciones de membresía (Default: None).
            input_dim (int): Dimensión de entrada a plotear. Si es None, se plotean todas las dimensiones de entrada (Default: None).
            group_by_dim (bool): Si es True, agrupa las funciones de membresía en un solo gráfico por cada dimensión de entrada. (Default: False)
        """
        variables = [f'x{i}' for i in range(self._input_size)]
        dataframe = self.premises_structure

        x = np.linspace(self._membership_function._max_val_plot, self._membership_function._min_val_plot, 500)

        # Determine which mfs and dimensions to plot
        if mf is not None:
            # Convert numeric index to string format if necessary
            if isinstance(mf, (int, float)):
                mf = f'MF {mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"MF '{mf}' not found in premises. Available mfs: {dataframe.index.tolist()}")
            mfs_to_plot = [mf]
        else:
            mfs_to_plot = dataframe.index

        # Validate input dimension
        if input_dim is not None:
            if not isinstance(input_dim, int) or input_dim < 0 or input_dim >= len(variables):
                raise ValueError(f"input_dim must be between 0 and {len(variables)-1}")
            dims_to_plot = [input_dim]
        else:
            dims_to_plot = range(len(variables))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(6, 4*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf in mfs_to_plot:
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, f'{param} ({var})']
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax.plot(x, y, label=f'{mf}')
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}')
                ax.grid(True)
                ax.set_xlabel('x')
                ax.set_ylabel('Membership Value')
                ax.legend()

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)
    
            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_mfs, ncols=n_dims, figsize=(5*n_dims, 4*n_mfs), squeeze=False)
    
            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = [dataframe.loc[mf, f'{param} ({var})'] for param in self._membership_function._params]
    
                        y = self._membership_function._simple_numpy_implementation(x, *params)
    
                        ax = axes[i, j]
                        ax.plot(x, y, label=f'{mf}, {var}')
                        ax.set_title(f'{mf}, {var}')
                        ax.grid(True)
                        if i == n_mfs - 1:
                            ax.set_xlabel('x')
                        if j == 0:
                            ax.set_ylabel('Membership Value')
                    except KeyError as e:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
            plt.tight_layout()
            plt.show()