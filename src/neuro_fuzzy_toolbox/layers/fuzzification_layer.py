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
    def __init__(self, mf_distribution, membership_function=GeneralizedBell_MF(), features=None, dtype=torch.float32):
        """
        Inicializa una nueva instancia de FuzzificationLayer.
        
        Args:
            mf_distribution (list): Lista de enteros que representan la cantidad de funciones de membresía para cada feature de entrada.
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            features (iterable): Iterable que contiene los nombres de las características de las variables de entrada como strings consideradas en el modelo (input features). Debe ser de largo input_size (Default: None).
            dtype (torch.dtype): Tipo de dato de los datos de entrada (Default: torch.float32).
        
        """
        super(FuzzificationLayer, self).__init__()
        
        # Input data info
        self._input_size = len(mf_distribution)
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(self._input_size)]
        if features != None and len(features) == self._input_size:
            self.features = features
        
        # Membership function
        self._membership_function = membership_function
        
        # Premise parameters
        self._max_n_mfs = max(mf_distribution)
        self._mf_distribution = torch.tensor(mf_distribution)
        
        premises = nn.ParameterList()
        for n_mfs in self._mf_distribution:
           premises.append(Parameter(self._membership_function.random_single_feature_mfs(n_mfs, dtype), requires_grad=True))
        self._premises = premises
        
    
    def forward(self, x):
        """
        Realiza un paso hacia adelante para calcular los valores de membresía para cada feature de entrada.
        
        Args:
            x (torch.Tensor): Conjunto de datos de entrada, tiene tamaño (batch_size, input_size).
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, input_size, max_n_mfs) que contiene los valores de membresía para cada feature, donde max_n_mfs es el número máximo de funciones de membresía entre todos los features.
            
        Note:
            Para manejar las distintas cantidades de funciones de membresía para cada feature, se rellenan con ceros las funciones de membresía faltantes. Esto podría ser mejorable en futuras versiones (agregando 0s directamente a un tensor que contenga los parámetros durante la instanciación de la capa, y no en el método forward).
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
        self._membership_function._max_val_plot = x_train.max().item()
        self._membership_function._min_val_plot = x_train.min().item()
    
    def get_premises(self):
        """
        Retorna los parámetros de las funciones de membresía de los antecedentes.
        
        Returns:
            list: Lista de parámetros de las funciones de membresía de los antecedentes.
        """
        return [mf.data.clone().detach() for mf in self._premises]
    
    
    def set_premises(self, premises):
        """
        Asigna los parámetros de las funciones de membresía de los antecedentes.
        
        Args:
            premises (list): Lista de tensores con los parámetros de las funciones de membresía. Cada tensor debe tener forma (num_mfs, mf_params), donde mf_params es el número de parámetros de la función de membresía.
        """
        for i, premise in enumerate(premises):
            self._premises[i] = Parameter(premise, requires_grad=True)
            
    
    def get_premises_as_parameters_list(self):
        """
        Retorna los parámetros de las funciones de membresía de los antecedentes como una lista de parámetros.
        
        Returns:
            nn.ParameterList: Lista de parámetros de las funciones de membresía de los antecedentes.
        """
        return self._premises
    
    
    @property
    def num_mfs(self):
        """
        Retorna la cantidad de funciones de membresía por feature.
        
        Returns:
            torch.tensor: Tensor con la cantidad de funciones de membresía por feature.
        """
        return self._mf_distribution
    
    
    @property
    def get_premises_structure(self):
        """
        Retorna la estructura de los antecedentes.
        
        Returns:
            pd.DataFrame: DataFrame de pandas que contiene la estructura de los antecedentes.
        """
        
        columns = pd.MultiIndex.from_product(
            [self.features, self._membership_function._params]
        )

        df = pd.DataFrame(columns=columns)

        mfs = [f'MF {i + 1}' for i in range(self._max_n_mfs)]

        for i in range(self._input_size):
            num_mfs = self._mf_distribution[i]
            for param_name in self._membership_function._params:
                column_data = [self._premises[i][mf_idx, self._membership_function._params.index(param_name)].item() 
                               if mf_idx < num_mfs else None 
                               for mf_idx in range(self._max_n_mfs)]

                df[(self.features[i], param_name)] = pd.Series(column_data, index=mfs)

        return df


    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False, linestyles='-', linewidths=2.5):
        """
        Plotea las funciones de membresía de los antecedentes.
        
        Args:
            mf (int): Función de membresía a plotear. Si es None, se plotean todas las funciones de membresía (Default: None).
            input_dim (int): Dimensión de entrada a plotear. Si es None, se plotean todas las dimensiones de entrada (Default: None).
            group_by_dim (bool): Si es True, agrupa las funciones de membresía en un solo gráfico por cada dimensión de entrada. (Default: False)
            linestyles (string | list): String o lista de strings que representan los tipos de linea para representar las funciones de membresía. Si se usa una lista, los tipos de linea se irán alternando. Los caracteres permitidos son: ['-', '--', '-.', ':']. (Default: '-')
            linewidths (float | list): Float o lista de floats que representan los grosores de linea para representar las funciones de membresía. Si se usa una lista, los grosores de linea se irán alternando. (Default: 2.5)
        """
        variables = self.features
        dataframe = self.get_premises_structure
        
        x = np.linspace(self._membership_function._min_val_plot, self._membership_function._max_val_plot, 500)
        
        # Determine which mfs and dimensions to plot
        if mf is not None:
            mf = f'{mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"'{mf}'index not found in premises. Available mfs: {dataframe.index.tolist()}")
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
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(mfs_to_plot)))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(10, 5*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf_idx, mf in enumerate(mfs_to_plot):
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            color = colors[mf_idx % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[mf_idx % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, label=f'{mf}', color=color, linestyle=linestyle, linewidth=linewidth)
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('Membership Value', fontsize=11)
                ax.legend(fontsize=10, loc='best')
                ax.set_ylim([0, 1.1])

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)

            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_dims, ncols=n_mfs, figsize=(8*n_mfs, 6*n_dims), squeeze=False)

            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax = axes[j, i] if n_mfs > 1 else axes[i, j]
                            color = colors[i % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[i % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{mf}, {var}')
                            ax.set_title(f'{mf}, {var}', fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim([0, 1.1])
                            if (i == n_mfs - 1) or (j == n_dims - 1):
                                ax.set_xlabel('x', fontsize=10)
                            if j == 0 or i == 0:
                                ax.set_ylabel('Membership Value', fontsize=10)
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
    def __init__(self, input_size, num_mfs=1, membership_function=GeneralizedBell_MF(), features=None, dtype=torch.float32):
        """
        Inicializa una nueva instancia de h_FuzzificationLayer.
        
        Args:
            input_size (int): Número de features de entrada.
            num_mfs (int): Número de funciones de membresía para cada feature (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            features (iterable): Iterable que contiene los nombres de las características de las variables de entrada como strings consideradas en el modelo (input features). Debe ser de largo input_size (Default: None).
            dtype (torch.dtype): Tipo de dato de los datos de entrada (Default: torch.float32).
        """
        super(h_FuzzificationLayer, self).__init__()

        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
        
        # Membership function
        self._membership_function = membership_function

        # Initialize premise parameters
        self._premises = Parameter(self._membership_function.random_premises(input_size, num_mfs, dtype), requires_grad=True)


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
        self._membership_function._max_val_plot = x_train.max().item()
        self._membership_function._min_val_plot = x_train.min().item()

    def get_premises(self):
        """
        Retorna los parámetros de las funciones de membresía de los antecedentes.
        
        Returns:
            torch.Tensor: Tensor con los parámetros de las funciones de membresía de los antecedentes.
        """
        return self._premises.data.clone().detach()
    
    
    def set_premises(self, premises):
        """
        Asigna los parámetros de las funciones de membresía de los antecedentes.
        
        Args:
            premises (torch.Tensor): Tensor con los parámetros de las funciones de membresía de los antecedentes.
        """
        self._premises = Parameter(premises, requires_grad=True)
        
        
    def get_premises_as_parameters_list(self):
        """
        Retorna los antecedentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            list: Lista de 1 solo elemento (nn.Parameter) que contiene los parámetros de los antecedentes.
        """
        return [self._premises]
    
    
    @property
    def num_mfs(self):
        """
        Retorna la cantidad de funciones de membresía por feature.
        
        Returns:
            int: Cantidad de funciones de membresía que tiene cada feature.
        """
        return self.get_premises().shape[1]
    

    @property
    def get_premises_structure(self):
        """
        Retorna la estructura de los antecedentes en un DataFrame de pandas.
        """
        premises = self.get_premises()
        
        mfs = [f"rule {i + 1}" for i in range(premises.shape[1])]

        columns = pd.MultiIndex.from_product(
            (self.features, self._membership_function._params),
        )

        df = pd.DataFrame(index=mfs, columns=columns)

        for i in range(self._input_size):
            for j, param in enumerate(self._membership_function._params):
                column_data = premises[i, :, j].tolist()
                df[(self.features[i], param)] = pd.Series(column_data, index=mfs)

        return df
    
    
    
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False, linestyles='-', linewidths=2.5):
        """
        Plotea las funciones de membresía de los antecedentes.
        
        Args:
            mf (int): Función de membresía a plotear. Si es None, se plotean todas las funciones de membresía (Default: None).
            input_dim (int): Dimensión de entrada a plotear. Si es None, se plotean todas las dimensiones de entrada (Default: None).
            group_by_dim (bool): Si es True, agrupa las funciones de membresía en un solo gráfico por cada dimensión de entrada. (Default: False)
            linestyles (string | list): String o lista de strings que representan los tipos de linea para representar las funciones de membresía. Si se usa una lista, los tipos de linea se irán alternando. Los caracteres permitidos son: ['-', '--', '-.', ':']. (Default: '-')
            linewidths (float | list): Float o lista de floats que representan los grosores de linea para representar las funciones de membresía. Si se usa una lista, los grosores de linea se irán alternando. (Default: 2.5)
        """
        variables = [f'{self.features[i]}' for i in range(self._input_size)]
        dataframe = self.get_premises_structure

        x = np.linspace(self._membership_function._min_val_plot, self._membership_function._max_val_plot, 500)

        # Determine which mfs and dimensions to plot
        if mf is not None:
            mf = f'{mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"'{mf}' index not found in premises. Available mfs: {dataframe.index.tolist()}")
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
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(mfs_to_plot)))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(10, 5*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf_idx, mf in enumerate(mfs_to_plot):
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            #value = dataframe.loc[mf, f'{param} ({var})']
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            color = colors[mf_idx % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[mf_idx % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, label=f'{mf}', color=color, linestyle=linestyle, linewidth=linewidth)
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('Membership Value', fontsize=11)
                ax.legend(fontsize=10, loc='best')
                ax.set_ylim([0, 1.1])

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)

            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_dims, ncols=n_mfs, figsize=(8*n_mfs, 6*n_dims), squeeze=False)

            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax = axes[j, i] if n_mfs > 1 else axes[i, j]
                            color = colors[i % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[i % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{mf}, {var}')
                            ax.set_title(f'{mf}, {var}', fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim([0, 1.1])
                            if (i == n_mfs - 1) or (j == n_dims - 1):
                                ax.set_xlabel('x', fontsize=10)
                            if j == 0 or i == 0:
                                ax.set_ylabel('Membership Value', fontsize=10)
                    except KeyError as e:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
            plt.tight_layout()
            plt.show()
            
            
    
class rule_reduced_FuzzificationLayer(nn.Module):
    """
    Clase para representar la capa de fuzzificación de un sistema de inferencia neuro-difuso adaptativo (ANFIS) con reducción de reglas, es decir, 
    con la misma cantidad de funciones de membresía para cada feature de los datos de entrada, limitando a que cada uno tenga el mismo 
    número de variables lingüisticas.
    """
    def __init__(self, input_size, num_mfs=1, membership_function=GeneralizedBell_MF(), features=None, dtype=torch.float32):
        """
        Inicializa una nueva instancia de rule_reduced_FuzzificationLayer.
        
        Args:
            input_size (int): Número de features de entrada.
            num_mfs (int): Número de funciones de membresía para cada feature (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            features (iterable): Iterable que contiene los nombres de las características de las variables de entrada como strings consideradas en el modelo (input features). Debe ser de largo input_size (Default: None).
            dtype (torch.dtype): Tipo de dato de los datos de entrada (Default: torch.float32).
        """
        super(rule_reduced_FuzzificationLayer, self).__init__()

        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
        
        # Membership function
        self._membership_function = membership_function

        # Initialize premise parameters
        self._premises = nn.ParameterList([
            nn.Parameter(premise, requires_grad=True) for premise in self._membership_function.random_premises(
                input_size=input_size, 
                num_mfs=num_mfs, 
                dtype=dtype
            ).unbind(1)
        ])


    def forward(self, x):
        """
        Realiza un paso hacia adelante para calcular los valores de membresía para cada feature de entrada.
        
        Args:
            x (torch.Tensor): Conjunto de datos de entrada, tiene tamaño (batch_size, input_size).
            
        Returns:
            torch.Tensor: Tensor de tamaño (batch_size, input_size, num_mfs) que contiene los valores de membresía para cada feature, donde num_mfs es el número de funciones de membresía para cada feature.
        """
        return self._membership_function(x, torch.stack([premise for premise in self._premises], 1))


    def init_premises(self, x_train):
        """
        Inicializa los parámetros de las funciones de membresía de los antecedentes utilizando los datos de entrenamiento.
        
        Args:
            x_train (torch.Tensor): Datos de entrenamiento de entrada.
        """
        self._dtype = x_train.dtype
        self._premises = nn.ParameterList([
            nn.Parameter(premise, requires_grad=True) for premise in self._membership_function.initialize_premises(x_train=x_train, num_mfs=len(self._premises)).unbind(1)
        ])
        self._membership_function._max_val_plot = x_train.max().item()
        self._membership_function._min_val_plot = x_train.min().item()
        
        
    def get_premises(self):
        """
        Retorna los antecedentes del modelo.
        
        Returns:
            torch.tensor: Tensor con los antecedentes del modelo. Su forma es (input_size, num_mfs, mf_params).
        """
        return torch.stack([premise.data.clone().detach() for premise in self._premises], 1)
        
        
    def set_premises(self, premises):
        """
        Setea los parámetros de las funciones de membresía de la capa de fuzzificación del modelo.
        
        Args:
            premises (torch.tensor): Tensor con los parámetros de las funciones de membresía. Su forma debe ser (input_size, num_mfs, mf_params), donde mf_params es el número de parámetros de la función de membresía.
        """
        self._premises = nn.ParameterList([
            nn.Parameter(premise, requires_grad=True) for premise in premises.unbind(1)
        ])
        
        
    def get_premises_as_parameters_list(self):
        """
        Retorna los antecedentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            nn.ParameterList: Lista de parámetros de las funciones de membresía de los antecedentes.
        """
        return self._premises
    
    
    @property
    def num_mfs(self):
        """
        Retorna la cantidad de funciones de membresía por feature.
        
        Returns:
            int: Cantidad de funciones de membresía que tiene cada feature.
        """
        return self.get_premises().shape[1]
        
    
    @property
    def get_premises_structure(self):
        """
        Retorna la estructura de los antecedentes en un DataFrame de pandas.
        """
        premises_tensor = torch.stack([premise.data.clone().detach() for premise in self._premises], 1)
        n_mfs = premises_tensor.shape[1]

        mfs = [f"rule {i + 1}" for i in range(n_mfs)]
        
        multi_columns = pd.MultiIndex.from_product(
            (self.features, self._membership_function._params)
        )

        df = pd.DataFrame(index=mfs, columns=multi_columns)

        for i in range(self._input_size):
            for j, param in enumerate(self._membership_function._params):
                column_data = premises_tensor[i, :, j].tolist()
                df.loc[:, (self.features[i], param)] = pd.Series(column_data, index=mfs)

        return df
    
    
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False, linestyles='-', linewidths=2.5):
        """
        Plotea las funciones de membresía de los antecedentes.
        
        Args:
            mf (int): Función de membresía a plotear. Si es None, se plotean todas las funciones de membresía (Default: None).
            input_dim (int): Dimensión de entrada a plotear. Si es None, se plotean todas las dimensiones de entrada (Default: None).
            group_by_dim (bool): Si es True, agrupa las funciones de membresía en un solo gráfico por cada dimensión de entrada. (Default: False)
            linestyles (string | list): String o lista de strings que representan los tipos de linea para representar las funciones de membresía. Si se usa una lista, los tipos de linea se irán alternando. Los caracteres permitidos son: ['-', '--', '-.', ':']. (Default: '-')
            linewidths (float | list): Float o lista de floats que representan los grosores de linea para representar las funciones de membresía. Si se usa una lista, los grosores de linea se irán alternando. (Default: 2.5)
        """
        variables = [f'{self.features[i]}' for i in range(self._input_size)]
        dataframe = self.get_premises_structure

        x = np.linspace(self._membership_function._min_val_plot, self._membership_function._max_val_plot, 500)

        # Determine which mfs and dimensions to plot
        if mf is not None:
            mf = f'{mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"'{mf}' index not found in premises. Available mfs: {dataframe.index.tolist()}")
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
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(mfs_to_plot)))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(10, 5*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf_idx, mf in enumerate(mfs_to_plot):
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            #value = dataframe.loc[mf, f'{param} ({var})']
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            color = colors[mf_idx % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[mf_idx % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, label=f'{mf}', color=color, linestyle=linestyle, linewidth=linewidth)
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('Membership Value', fontsize=11)
                ax.legend(fontsize=10, loc='best')
                ax.set_ylim([0, 1.1])

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)

            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_dims, ncols=n_mfs, figsize=(8*n_mfs, 6*n_dims), squeeze=False)

            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax = axes[j, i] if n_mfs > 1 else axes[i, j]
                            color = colors[i % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[i % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{mf}, {var}')
                            ax.set_title(f'{mf}, {var}', fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim([0, 1.1])
                            if (i == n_mfs - 1) or (j == n_dims - 1):
                                ax.set_xlabel('x', fontsize=10)
                            if j == 0 or i == 0:
                                ax.set_ylabel('Membership Value', fontsize=10)
                    except KeyError as e:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
            plt.tight_layout()
            plt.show()