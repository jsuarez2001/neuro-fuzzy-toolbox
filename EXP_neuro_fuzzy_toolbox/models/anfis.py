import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from EXP_neuro_fuzzy_toolbox.func import GeneralizedBell_MF, Linear_CF
from EXP_neuro_fuzzy_toolbox.layers import (
    h_FuzzificationLayer, 
    h_FiringLevelsLayer, 
    FuzzificationLayer,
    rule_reduced_FuzzificationLayer,
    FiringLevelsLayer,
    NormalizationLayer, 
    ConsequentLayer,
    alt_ConsequentLayer,
    OutputLayer
    )

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

import itertools


class base_ANFIS(nn.Module):
    """
    Clase base para un sistema de inferencia neuro-difuso adaptativo (ANFIS). Esta clase contiene los métodos y atributos comunes en los modelos ANFIS implementados en el toolbox. 
    Las clases ANFIS y h_ANFIS heredan de esta clase y añaden funcionalidades específicas.
    
    Warning:
        Esta clase no debe ser instanciada directamente.
    
    """
    
    def forward(self, x, return_probabilities=False):
        """
        Realiza un paso hacia adelante a través del modelo.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            return_probabilities (bool): Indica si el resultado pasará por una función Softmax para obtener probabilidades. Solo se aplica si el tipo de salida es 'softmax', en caso contrario, se ignora (Default: False).
        
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
        
    
    def init_consequents(self, x, y):
        """
        Inicializa los parámetros consecuentes del modelo usando una estimación de mínimos cuadrados.
        
        Note:
            Específicamente, se usa la descomposición QR con pivoteo para resolver el problema de mínimos cuadrados. Para más información, ver: https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html.
        
        Args:
            x (torch.Tensor): Tensor con los datos de entrada. Es de tamaño (batch_size, input_size).
            y (torch.Tensor): Tensor con los datos de salida. Es de tamaño (batch_size, outputs).
        """
        _, w_norm, _ = self.intermediate_values(x)
        xe = torch.cat([x, torch.ones(x.shape[0], 1, dtype=self._dtype)], dim=1)
        fs = w_norm.unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
        X = xe.repeat(1, self.rules)
        
        '''preliminary fix for the dtype issue'''
        if self._output_type == 'softmax':
            y = torch.nn.functional.one_hot(y, self._outputs)
        if y.dtype != X.dtype:
            y = y.to(X.dtype)
        '''preliminary fix for the dtype issue'''
        
        # Solve least squares problem using QR decomposition with pivoting
        C, _, _, _ = torch.linalg.lstsq(X * fs, y)
        new_consequents = C.t().reshape(self._outputs, self.rules, xe.shape[1])
        
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
        
        if self._output_type == 'softmax':
            with torch.no_grad():
                output = self.forward(x, return_probabilities=True)
            output = torch.argmax(output, dim=1).detach().numpy()
            
        elif self._output_type == 'sigmoid':
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
    def num_mfs(self):
        """
        Retorna la cantidad de funciones de membresía por feature.
        
        Returns:
            int: Cantidad de funciones de membresía que tiene cada feature.
        """
        return self._fuzzification_layer.num_mfs
    
    @property
    def rules(self):
        """
        Retorna la cantidad de reglas del modelo ANFIS.
        
        Returns:
            int: Cantidad de reglas.
        """
        return self.get_consequents().shape[1]
    
    
    # ----- Premises seters and getters -----
    def get_premises(self):
        """
        Retorna los antecedentes del modelo.
        
        Returns:
            torch.tensor: Tensor con los antecedentes del modelo. Su forma es (input_size, num_mfs, mf_params).
        """
        return self._fuzzification_layer.get_premises()
    
    def set_premises(self, premises):
        """
        Setea los parámetros de las funciones de membresía de la capa de fuzzificación del modelo.
        
        Args:
            premises (torch.tensor): Tensor con los parámetros de las funciones de membresía. Su forma debe ser (input_size, num_mfs, mf_params), donde mf_params es el número de parámetros de la función de membresía.
        """
        self._fuzzification_layer.set_premises(premises)
    
    def get_premises_as_parameters_list(self):
        """
        Retorna los antecedentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            list: Lista de 1 solo elemento (nn.Parameter) que contiene los parámetros de los antecedentes.
        """
        return self._fuzzification_layer.get_premises_as_parameters_list()
    
    
    # ----- Consequents seters and getters -----
    def set_consequents(self, consequents):
        """
        Setea los parámetros consecuentes del modelo.
        
        Args:
            consequents (torch.tensor): Tensor con los parámetros consecuentes. Su forma debe ser (outputs, rules, input_size + 1).
        """
        self._consequent_layer.set_consequents(consequents)
    
    def get_consequents(self):
        """
        Retorna los parámetros consecuentes del modelo.
        
        Returns:
            torch.tensor: Tensor con los parámetros consecuentes. Su forma es (outputs, rules, input_size + 1).
        """
        return self._consequent_layer.get_consequents()
    
    def get_consequents_as_parameters_list(self):
        """
        Retorna los parámetros consecuentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            list: Lista de 1 solo elemento (nn.Parameter) que contiene los parámetros consecuentes.
        """
        return self._consequent_layer.get_consequents_as_parameters_list()
    
    
    # ----- Parameters dataframes -----
    def get_premises_structure(self):
        """
        Retorna la estructura de los parámetros de las funciones de membresía.
        
        Returns:
            pandas.DataFrame: DataFrame con la estructura de los parámetros de las funciones de membresía.
        """
        return self._fuzzification_layer.premises_structure
    
    def get_consequents_structure(self):
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
    El procedimiento en este caso se realizaría solo multiplicando entre sí los valores de pertenencia *i* de cada feature, dando como resultado una cantidad de reglas igual al número de funciones de membresía de cada feature. (esto se detalla de mejor manera en :ref:`rule-reduced ANFIS <rule-reduced ANFIS>`).
    """
    
    def __init__(self, input_size, num_mfs, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, output_type="default", rule_reduced=False, linguistic_variables=None, dtype=torch.float32):
        """
        Inicializa un modelo ANFIS homogéneo.
        
        Args:
            input_size (int): Número de features de los datos de entrada.
            num_mfs (int): Número de funciones de membresía por feature.
            outputs (int): Número de salidas del modelo (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            consequent_function (ConsequentFunction): Función consecuente a utilizar (Default: Linear_CF).
            output_type (str): Tipo de salida del modelo (Default: 'default').
            rule_reduced (bool): True si se desea instanciar un ANFIS de reglas reducidas, False en caso contrario (Default: False).
            linguistic_variables (iterable): Iterable que contiene las variables lingüisticas como strings consideradas en el modelo (input features). Debe ser de largo input_size (Default: None).
            dtype (torch.dtype): Tipo de dato a utilizar en el modelo (Default: torch.float32).
        """
        super(h_ANFIS, self).__init__()
        
        if rule_reduced:
            rules = num_mfs
        else:
            rules = num_mfs**input_size
        
        
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.linguistic_variables = [f"x{i}" for i in range(input_size)]
        if linguistic_variables != None and len(linguistic_variables) == input_size:
            self.linguistic_variables = linguistic_variables
        
        
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
    
    
    # ----- Load state dict -----
    def load_state_dict(self, state_dict):
        """
        Carga un estado del modelo.
        
        Args:
            state_dict (dict): Diccionario con el estado del modelo.
        """
        self.set_premises(state_dict['_fuzzification_layer._premises'])
        self.set_consequents(state_dict['_consequent_layer._consequents'])
        
        
    # ----- Rules -----
    def get_rules_structure(self):
        premises_df = self.get_premises_structure()
        
        variables = premises_df.columns.get_level_values(0).unique()
        params = premises_df.columns.get_level_values(1).unique()
        idxs = premises_df.index.tolist()
        
        combs = itertools.product(idxs, repeat=len(variables))
        
        records = []
        for comb in combs:
            row = {}
            for var, sel in zip(variables, comb):
                for param in params:
                    row[(var, param)] = premises_df.loc[sel, (var, param)]
            records.append(row)

        premises_df = pd.DataFrame.from_records(records)
        premises_df.columns = pd.MultiIndex.from_tuples(premises_df.columns)
        premises_df = premises_df[premises_df.columns]
        premises_df.index = [f"rule {i+1}" for i in range(len(premises_df))]
        
        premises_df.columns = pd.MultiIndex.from_tuples(
            [("premises", *col) for col in premises_df.columns]
        )
        
        rules_dfs = self.get_consequents_structure()
        for i, df in enumerate(rules_dfs):
            df.columns = pd.MultiIndex.from_tuples(
                [(f"output {i+1} consequents", *col) for col in df.columns]
            )

        rules_dfs.insert(0, premises_df)

        return pd.concat(rules_dfs, axis=1)
        
        

class ANFIS(base_ANFIS):
    """
    Clase para un sistema de inferencia neuro-difuso adaptativo (ANFIS) con una cantidad de funciones de membresía arbitraria para cada feature de los datos de entrada.
    """
    
    def __init__(self, mf_distribution, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, output_type="default", linguistic_variables=None, dtype=torch.float32):
        """
        Inicializa un modelo ANFIS.
        
        Args:
            mf_distribution (list): Lista con la cantidad de funciones de membresía por feature de los datos de entrada.
            outputs (int): Número de salidas del modelo (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            consequent_function (ConsequentFunction): Función consecuente a utilizar (Default: Linear_CF).
            output_type (str): Tipo de salida del modelo (Default: 'default').
            dtype (torch.dtype): Tipo de dato a utilizar en el modelo (Default: torch.float32).
            
        """
        super(ANFIS, self).__init__()
        # Input data info
        self._input_size = len(mf_distribution)
        self._dtype = dtype
        self.linguistic_variables = [f"x{i}" for i in range(self._input_size)]
        if linguistic_variables != None and len(linguistic_variables) == self._input_size:
            self.linguistic_variables = linguistic_variables
        
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
        return super().get_premises()
    
    def set_premises(self, premises):
        """
        Setea los parámetros de las funciones de membresía de la capa de fuzzificación del modelo.
        
        Args:
            premises (list): Lista de tensores con los parámetros de las funciones de membresía. Cada tensor debe tener forma (num_mfs, mf_params), donde mf_params es el número de parámetros de la función de membresía.
        """
        super().set_premises(premises)
            
    def get_premises_as_parameters_list(self):
        """
        Retorna los antecedentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            nn.ParameterList: Lista de parámetros que contiene los antecedentes.
        """
        return super().get_premises_as_parameters_list()


    # ---- ANFIS parameters info ----
    @property
    def num_mfs(self):
        """
        Retorna la cantidad de funciones de membresía por feature.
        
        Returns:
            torch.tensor: Tensor con la cantidad de funciones de membresía por feature.
        """
        super().num_mfs
    
    
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
        
        
    # ----- Rules -----
    def get_rules_structure(self):
        premises_df = self.get_premises_structure()
        
        variables = premises_df.columns.get_level_values(0).unique()
        params = premises_df.columns.get_level_values(1).unique()

        mf_dist = self._mf_distribution.tolist()

        combs = itertools.product(*(range(n) for n in mf_dist))

        records = []
        for comb in combs:
            row = {}
            for var, sel in zip(variables, comb):
                for param in params:
                    row[(var, param)] = premises_df.iloc[sel][(var, param)]
            records.append(row)

        premises_df = pd.DataFrame.from_records(records)
        premises_df.columns = pd.MultiIndex.from_tuples(premises_df.columns)
        premises_df = premises_df[premises_df.columns]
        premises_df.index = [f"rule {i+1}" for i in range(len(premises_df))]

        premises_df.columns = pd.MultiIndex.from_tuples(
            [("premises", *col) for col in premises_df.columns]
        )

        rules_dfs = self.get_consequents_structure()

        for i, df in enumerate(rules_dfs):
            df.index = premises_df.index
            df.columns = pd.MultiIndex.from_tuples(
                [(f"output {i+1} consequents", *col) for col in df.columns]
            )

        rules_dfs.insert(0, premises_df)
        return pd.concat(rules_dfs, axis=1)
    

    
class rule_reduced_ANFIS(base_ANFIS):
    """
    Clase para un sistema de inferencia neuro-difuso adaptativo (ANFIS) homogéneo, es decir, con la misma cantidad de funciones de membresía para cada feature de los datos de entrada, 
    limitando a que cada uno con el mismo número de variables lingüisticas. Tiene la particularidad de que el número de reglas generadas en el cálculo de los niveles de disparo es reducido. 
    En vez de hacer la combinatoria completa para las multiplicaciones de los valores de pertenencia, el procedimiento se realizaría solo multiplicando entre sí los valores de pertenencia *i* 
    entre los features, dando como resultado una cantidad de reglas igual al número de funciones de membresía (en cada feature). Esto se detalla de mejor manera en :ref:`rule-reduced ANFIS <rule-reduced ANFIS>`.
    """
    
    def __init__(self, input_size, num_mfs, outputs=1, membership_function=GeneralizedBell_MF, consequent_function=Linear_CF, output_type="default", linguistic_variables=None, dtype=torch.float32):
        """
        Inicializa un modelo ANFIS homogéneo.
        
        Args:
            input_size (int): Número de features de los datos de entrada.
            num_mfs (int): Número de funciones de membresía por feature.
            outputs (int): Número de salidas del modelo (Default: 1).
            membership_function (MembershipFunction): Función de membresía a utilizar (Default: GeneralizedBell_MF).
            consequent_function (ConsequentFunction): Función consecuente a utilizar (Default: Linear_CF).
            output_type (str): Tipo de salida del modelo (Default: 'default').
            rule_reduced (bool): True si se desea instanciar un ANFIS de reglas reducidas, False en caso contrario (Default: False).
            dtype (torch.dtype): Tipo de dato a utilizar en el modelo (Default: torch.float32).
        """
        super(rule_reduced_ANFIS, self).__init__()
        self._rule_reduced = True
        
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.linguistic_variables = [f"x{i}" for i in range(input_size)]
        if linguistic_variables != None and len(linguistic_variables) == input_size:
            self.linguistic_variables = linguistic_variables
            
        
        # Output info
        self._output_type = output_type
        self._outputs = outputs
        
        
        # Layers
        mf_distribution = [num_mfs] * input_size
        self._fuzzification_layer = rule_reduced_FuzzificationLayer(
            input_size=input_size,
            num_mfs=num_mfs,
            membership_function=membership_function, 
            dtype=dtype
            )
        
        self._firing_levels_layer = h_FiringLevelsLayer(rule_reduced=True)
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = alt_ConsequentLayer(
            input_size=input_size,
            rules=num_mfs,
            outputs=outputs,
            consequent_function=consequent_function, 
            dtype=dtype
            )
        
        self._output_layer = OutputLayer(output_type=self._output_type)
        
        
    # ----- Premises seters and getters -----
    def get_premises_as_parameters_list(self):
        """
        Retorna los antecedentes del modelo como una lista de parámetros. Esto es útil para algoritmos de optimización.
        
        Returns:
            nn.ParameterList: Lista de parámetros de las funciones de membresía de los antecedentes.
        """
        return super().get_premises_as_parameters_list()
    
    
    # ----- Load state dict -----
    def load_state_dict(self, state_dict):
        """
        Carga un estado del modelo.
        
        Args:
            state_dict (dict): Diccionario con el estado del modelo.
        """
        prems = []
        cons = []
        for element in state_dict:
            if 'premises' in element:
                prems.append(state_dict[element])
            else:
                cons.append(state_dict[element])
        self.set_premises(torch.stack(prems, dim=1))
        self.set_consequents(torch.stack(cons, dim=1))
    
    
    # ----- Rules -----
    def get_rules_structure(self):
        premises_df = self.get_premises_structure()
        premises_df.index = [f"rule {i+1}" for i in range(len(premises_df))]
        
        premises_df.columns = pd.MultiIndex.from_tuples(
            [("premises", *col) for col in premises_df.columns]
        )
        
        rules_dfs = self.get_consequents_structure()
        for i, df in enumerate(rules_dfs):
            df.columns = pd.MultiIndex.from_tuples(
                [(f"output {i+1} consequents", *col) for col in df.columns]
            )

        rules_dfs.insert(0, premises_df)

        return pd.concat(rules_dfs, axis=1)
    
        
    
    """
    # ----- UUUGLY METHOD -----
    def show_rule(self, rule, ncols=1, figsize=(6,4)):
        x_vals = np.linspace(
            self._fuzzification_layer._membership_function._min_val_plot, 
            self._fuzzification_layer._membership_function._max_val_plot, 
            500
        )
        
        nrows = torch.ceil(torch.tensor(self._input_size/ncols)).int().item() + 1
        
        fig = plt.figure(figsize=(figsize[0] * ncols, figsize[1] * nrows))
        gs = gridspec.GridSpec(nrows, ncols, hspace=0.5)
        
        axes = []
        for r in range(nrows):
            for c in range(ncols):
                ax = fig.add_subplot(gs[r, c])
                axes.append(ax)
                
        for j in range(self._input_size, nrows * ncols):
            axes[j].axis('off')

        for j in range(self._input_size):
            params = []
            for param in self._fuzzification_layer._membership_function._params:
                value = self.premises_structure.loc[f'{rule-1}', (f'x{j}', f'{param}')]
                if pd.isna(value):
                    break
                params.append(value)

            y = self._fuzzification_layer._membership_function._simple_numpy_implementation(x_vals, *params)
            
            formatted_params = ", ".join([
                f"{self._fuzzification_layer._membership_function._params[i]}: {params[i]:.2f}" 
                for i in range(len(params))
                ])
                
            axes[j].plot(x_vals, y, label=f"{formatted_params}")
            axes[j].set_title(f"x{j}: {self.linguistic_variables[j]}")
            axes[j].set_xlabel(f"x{j}")
            axes[j].legend()
            axes[j].grid(True)
            #axes[j].axis('off')
            #axes[j].set_frame_on(False)
            
        ax_out = fig.add_subplot(gs[nrows - 1, :])
        ax_out.axis('off')
        
        output_text = ""
        for idx, df in enumerate(self.consequents_structure, start=1):
            try:
                row_values = df.loc[f"rule {rule}"]
            except KeyError:
                row_values = df.iloc[rule-1]
            
            ecuation = []
            for i, val in enumerate(row_values):
                if i == 0:
                    if val < 0:
                        ecuation.append(r"$-$")
                        ecuation.append(f"{-val:.2f}x{i}")
                    else:
                        ecuation.append("   ")
                        ecuation.append(f"{val:.2f}x{i}")
                else:
                    if val > 0:
                        ecuation.append(r"$+$")
                        ecuation.append(f"{val:.2f}x{i}")
                    else:
                        ecuation.append(r"$-$")
                        ecuation.append(f"{-val:.2f}x{i}")
                    
            formatted_values = " ".join(ecuation)
            output_text += f"Output{idx}: {formatted_values}\n"

        ax_out.text(0.5, 0.5, output_text, ha='center', va='center', fontsize=12)
        
        plt.show()
    """