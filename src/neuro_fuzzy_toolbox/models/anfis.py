import torch
import torch.nn as nn

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF
from neuro_fuzzy_toolbox.layers import (
    FuzzificationLayer,
    h_FuzzificationLayer, 
    rule_reduced_FuzzificationLayer,
    
    FiringLevelsLayer,
    h_FiringLevelsLayer,
    rule_reduced_FiringLevelsLayer,
    
    NormalizationLayer, 
    
    ConsequentLayer,
    alt_ConsequentLayer,
    
    OutputLayer
    )

import pandas as pd

import itertools


class base_ANFIS(nn.Module):
    """
    Base class for an Adaptive Neuro-Fuzzy Inference System (ANFIS).
    
    Contains the common methods and attributes shared across the ANFIS model
    variants implemented in this toolbox. The ANFIS, h_ANFIS and rule_reduced_ANFIS classes inherit
    from this class and extend it with variant-specific functionality.
    
    Warning:
        This class should not be instantiated directly.
    """
    
    def forward(self, x, return_probs=False):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input data tensor of shape ``(batch_size, input_size)``.
            return_probs (bool): If ``True``, the output is passed through a Softmax
                function to obtain class probabilities. Only applies when the model's
                output type is ``'softmax'``; ignored otherwise. Defaults to ``False``.
        """
        output = self._fuzzification_layer(x)
        output = self._consequent_layer(x, self._normalization_layer(self._firing_levels_layer(output)))
        output = self._output_layer(output, return_probs)
        return output
    
    
    # ---- Initialize parameters ----
    def init_premises(self, x):
        """
        Initializes the premise parameters of the model's fuzzification layer from the provided data.
        
        Args:
            x (torch.Tensor): Input data tensor of shape ``(batch_size, input_size)``.
        """
        self._dtype = x.dtype
        self._consequent_layer._to_dtype(x.dtype) # Set dtype to consequents
        self._fuzzification_layer.init_premises(x)
        
    
    def init_consequents(self, x, y, driver=None, ridge_lambda=0.):
        """
        Initializes the consequent parameters of the model using a least-squares estimate.
        
        Note:
            The backend method used for the least-squares estimation is specified by the
            ``driver`` parameter. For more information, see:
            https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html.
        
        Args:
            x (torch.Tensor): Input data tensor of shape ``(batch_size, input_size)``.
            y (torch.Tensor): Output data tensor of shape ``(batch_size, outputs)``.
            driver (str): Backend function to use for the least-squares estimation. Valid values are
                ``'gels'``, ``'gelsy'``, ``'gelsd'``, and ``'gelss'``. If ``None``, defaults to ``'gels'``.
            ridge_lambda (float): Lambda value for Ridge regularization in the least-squares estimation.
                If ``0.``, no regularization is applied. Defaults to ``0.``.
        
        Important:
            If the model has ``output_type='softmax'``, the class labels in ``y`` are expected to be integers representing
            the target classes, and a one-hot encoding of these labels will be performed internally for the least-squares
            estimation. If the labels are not of the form ``[0, 1, 2, ...]``, the model will automatically adjust to the labels
            present in ``y`` and set them as the classes it will attempt to predict. This is useful when users prefer to work
            with custom class labels directly. The target class labels can also be set manually using :meth:`set_custom_classes_ids`.
        """
        w_norm = self.get_firing_levels(x, normalized=True)
        xe = torch.cat([x, torch.ones(x.shape[0], 1, dtype=self._dtype)], dim=1)
        fs = w_norm.unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
        X = xe.repeat(1, self.rules)
        
        '''preliminary fix for the dtype issue'''
        if self._output_type == 'softmax':
            if not torch.equal(torch.unique(y), torch.arange(self._outputs)):
                y = torch.searchsorted(torch.unique(y), y)
                if not self._custom_classes:
                    self.set_custom_classes_ids(torch.unique(y))
                    print(f"Custom classes set to: {self._classes}")
            y = torch.nn.functional.one_hot(y, self._outputs)
        if y.dtype != X.dtype:
            y = y.to(X.dtype)
        '''preliminary fix for the dtype issue'''
        
        A = X * fs
        
        if ridge_lambda > 0.:
            p = A.shape[1]
            I = torch.eye(p, dtype=A.dtype) * torch.sqrt(torch.tensor(ridge_lambda, dtype=A.dtype))
            A = torch.cat([A, I], dim=0)
            if y.dim() > 1:
                m = y.shape[1]
                zeros = torch.zeros((p, m), dtype=A.dtype)
            else:
                zeros = torch.zeros(p, dtype=A.dtype)
            y  = torch.cat([y, zeros], dim=0)
        
        # Solve least squares problem using QR decomposition with pivoting
        C, _, _, _ = torch.linalg.lstsq(A, y, driver=driver)
        new_consequents = C.t().reshape(self._outputs, self.rules, xe.shape[1])
        
        self.set_consequents(new_consequents)
        
    
    # ---- Model predict ----
    def predict(self, x):
        """
        Runs inference on the input data and adjusts the output to the expected format based on the model's output type.
        
        Args:
            x (torch.Tensor): Input data tensor of shape ``(batch_size, input_size)``.
            
        Returns:
            torch.Tensor: Model predictions.
        """
        if self._output_type == 'default':
            with torch.no_grad():
                output = self.forward(x).detach()
                
        elif self._output_type == 'sigmoid':
            with torch.no_grad():
                output = self.forward(x).detach()
            output = (output > 0.5).to(int)
        
        elif self._output_type == 'softmax':
            with torch.no_grad():
                output = self.forward(x, return_probs=True)
            output = torch.argmax(output, dim=1).detach()
            
            if self._custom_classes:
                output = self._classes[output]
            
            """
            if not torch.equal(self.classes, torch.arange(self._outputs)):
                output = self.classes[output].numpy()
            else:
                output = output.numpy()
            """
            
        return output
    
    @property
    def classes(self):
        """
        Returns the class labels that the model attempts to predict.

        Returns:
            torch.Tensor: Tensor containing the class labels that the model attempts to predict.
        """
        return self._classes
    
    def set_custom_classes_ids(self, new_classes_ids):
        """
        Sets custom labels for the classes that the model attempts to predict.

        Note:
            By default, for a problem with ``C`` classes, the model uses labels of
            the form ``[0, 1, ..., C-1]``. This method allows setting custom class
            labels, which will always be stored in ascending order.

        Important:
            Only applicable when the model has ``output_type='softmax'``.

        Args:
            new_classes_ids (list[int]): List containing the new class labels.
        """
        outputs = self._outputs
        if len(new_classes_ids) != outputs:
            raise ValueError(f"Provided list of length {len(new_classes_ids)} does not match the number of classes: {self._outputs}")
        
        self._classes = torch.tensor(new_classes_ids, dtype=torch.long).sort().values
        
        if torch.equal(self._classes, torch.arange(outputs)):
            self._custom_classes = False
        else:
            self._custom_classes = True
        print(f"New classes: {self._classes}")
        
    
    # ---- Intermediate values ----
    def get_firing_levels(self, x, normalized=False):
        """
        Returns the firing levels of the model for the given input data.
        
        Args:
            x (torch.Tensor): Input data tensor of shape ``(batch_size, input_size)``.
            normalized (bool): If ``True``, returns the normalized firing levels. Defaults to ``False``.

        Returns:
            torch.Tensor: Firing levels of shape ``(batch_size, num_rules)``.
        """
        with torch.no_grad():
            w = self._fuzzification_layer(x)
            w = self._firing_levels_layer(w)
            if normalized:
                w = self._normalization_layer(w)
        return w
    
    def get_all_consequents_outputs(self, x, weighted=True):
        """
        Returns the individual rule outputs of the model for the given input data.
        
        Args:
            x (torch.Tensor): Input data tensor of shape ``(batch_size, input_size)``.
            weighted (bool): If ``True``, the rule outputs are weighted by their corresponding firing levels. Defaults to ``True``.
            
        Returns:
            torch.Tensor: Individual rule outputs of shape ``(outputs, batch_size, num_rules)``.
        """
        if weighted:
            with torch.no_grad():
                w = self._fuzzification_layer(x)
                w = self._firing_levels_layer(w)
                w_norm = self._normalization_layer(w)
                outputs = self._consequent_layer(x, w_norm)
        else:
            outputs = self._consequent_layer.get_consequents_outputs(x)
        return outputs


    # ---- ANFIS structure info ----
    @property
    def num_mfs(self):
        """
        Returns the number of membership functions per input feature.
        
        Returns:
            int: Number of membership functions per input feature.
        """
        return self._fuzzification_layer.num_mfs
    
    @property
    def rules(self):
        """
        Returns the number of rules in the ANFIS model.

        Returns:
            int: Number of rules.
        """
        return self.get_consequents().shape[1]
    
    @property
    def outputs(self):
        """
        Returns the number of outputs of the model.

        Returns:
            int: Number of outputs.
        """
        return self.get_consequents().shape[0]
    
    
    # ----- Premises seters and getters -----
    def get_premises(self):
        """
        Returns the premise parameters of the model.

        Returns:
            torch.Tensor: Tensor containing the premise parameters of shape
            ``(input_size, num_mfs, mf_params)``.
        """
        return self._fuzzification_layer.get_premises()
    
    def set_premises(self, premises):
        """
        Sets the membership function parameters of the model's fuzzification layer.

        Args:
            premises (torch.Tensor): Tensor containing the premise parameters of shape ``(input_size, num_mfs, mf_params)``,
                where ``mf_params`` is the number of parameters of the membership function.
        """
        self._fuzzification_layer.set_premises(premises)
    
    def get_premises_as_parameters_list(self):
        """
        Returns the premise parameters of the model as a list of parameters.
        This is useful for optimization algorithms (using PyTorch optimizers).
    
        Returns:
            list[nn.Parameter]: A list containing a single element (``nn.Parameter``) with the
            premise parameters.
        """
        return self._fuzzification_layer.get_premises_as_parameters_list()
    
    
    # ----- Consequents seters and getters -----
    def set_consequents(self, consequents):
        """
        Sets the consequent parameters of the model.

        Args:
            consequents (torch.Tensor): Tensor containing the consequent parameters of shape ``(outputs, rules, input_size + 1)``.
        """
        self._consequent_layer.set_consequents(consequents)
    
    def get_consequents(self):
        """
        Returns the consequent parameters of the model.

        Returns:
            torch.Tensor: Tensor containing the consequent parameters of shape ``(outputs, rules, input_size + 1)``.
        """
        return self._consequent_layer.get_consequents()
    
    def get_consequents_as_parameters_list(self):
        """
        Returns the consequent parameters of the model as a list of parameters.
        This is useful for optimization algorithms (using PyTorch optimizers).

        Returns:
            list[nn.Parameter]: A list containing a single element (``nn.Parameter``) with the consequent parameters.
        """
        return self._consequent_layer.get_consequents_as_parameters_list()
    
    
    # ----- Parameters dataframes -----
    def get_premises_structure(self):
        """
        Returns the structure of the premise parameters.

        Returns:
            pandas.DataFrame: DataFrame containing the structure of the premise
            parameters.
        """
        return self._fuzzification_layer.get_premises_structure
    
    def get_consequents_structure(self):
        """
        Returns the structure of the consequent parameters.

        Returns:
            list[pandas.DataFrame]: A list of pandas DataFrames containing the structure of the
            consequent parameters.
        """
        return self._consequent_layer.get_consequents_structure
    
    
    # ----- Plot premises -----
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False, linestyles='-', linewidths=2.5):
        """
        Plots the membership functions of the model's premises.
        
        Args:
            mf (int): Index of the membership function to plot. If ``None``, all membership functions are plotted. Defaults to ``None``.
            input_dim (int): Input feature dimension to plot. If ``None``, all dimensions are plotted. Defaults to ``None``.
            group_by_dim (bool): If ``True``, groups all membership functions into a single plot per input dimension. Defaults to ``False``.
            linestyles (str | list[str]): A string or list of strings specifying the line styles used to represent the membership functions. 
                If a list is provided, the styles are applied cyclically. Valid values are: ``'-'``, ``'--'``, ``'-.'``, ``':'``. 
                Defaults to ``'-'``.
            linewidths (float): Line width used to plot the membership functions. Defaults to ``2.5``.
        """
        self._fuzzification_layer.plot_premises(mf, input_dim, group_by_dim, linestyles, linewidths)
    


class h_ANFIS(base_ANFIS):
    """
    Homogeneous Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Implements an ANFIS model where every input feature shares the same number
    of membership functions, restricting each feature to the same number of
    linguistic variables.

    Supports an optional rule-reduced mode that avoids the full combinatorial
    expansion when computing firing levels. In this mode, only the membership
    degrees at matching indices across features are multiplied together, yielding
    a number of rules equal to the number of membership functions per feature
    rather than ``num_mfs ** input_size``. For further details, see
    :ref:`rule-reduced ANFIS <rule-reduced ANFIS>`.
    """
    
    def __init__(self, input_size, num_mfs, outputs=1, membership_function=GeneralizedBell_MF(), output_type="default", rule_reduced=False, features=None, dtype=torch.float32):
        """
        Initializes a homogeneous ANFIS model.

        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of membership functions per input feature.
            outputs (int): Number of model outputs. Defaults to ``1``.
            membership_function (MembershipFunction): Membership function to use. Defaults to ``GeneralizedBell_MF``.
            output_type (str): Output type of the model. Defaults to ``'default'``.
            rule_reduced (bool): If ``True``, instantiates a rule-reduced ANFIS model. Defaults to ``False``.
            features (iterable): Iterable of strings containing the names of the input features considered by the model.
                Must be of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type to use in the model. Defaults to ``torch.float32``.
        """
        super(h_ANFIS, self).__init__()
        if rule_reduced:
            rules = num_mfs
        else:
            rules = num_mfs**input_size
        
        
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
        
        
        # ANFIS structure info
        self._rule_reduced = rule_reduced
        
        
        # Output info
        self._output_type = output_type
        self._outputs = outputs
        self._classes = torch.arange(outputs)
        self._custom_classes = False
        
        
        # Layers
        self._fuzzification_layer = h_FuzzificationLayer(
            input_size=input_size,
            num_mfs=num_mfs, 
            membership_function=membership_function,
            features=features,
            dtype=dtype
            )
        
        self._firing_levels_layer = h_FiringLevelsLayer(rule_reduced=rule_reduced)
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = ConsequentLayer(
            input_size=input_size,
            rules=rules,
            outputs=outputs,
            features=features,
            dtype=dtype
            )
        
        self._output_layer = OutputLayer(output_type=self._output_type)
    
    
    # ----- Load state dict -----
    def load_state_dict(self, state_dict):
        """
        Loads a model state dictionary.

        Args:
            state_dict (dict): Dictionary containing the model state.
        """
        self.set_premises(state_dict['_fuzzification_layer._premises'])
        self.set_consequents(state_dict['_consequent_layer._consequents'])
        
        
    # ----- Rules -----
    def get_rules_structure(self):
        """
        Returns a combined summary of the premises and consequent parameters for
        each rule in the model.

        The resulting DataFrame organizes each rule as a row, with columns grouped
        first by premises (showing the membership function parameters of each input
        feature) and then by the consequent parameters of each model output.

        Returns:
            pandas.DataFrame: DataFrame with a MultiIndex column structure, where the top-level groups correspond to ``'premises'`` and
            ``'output i consequents'`` for each output ``i``, and rows correspond to individual rules.
        """
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
    Adaptive Neuro-Fuzzy Inference System (ANFIS) with an arbitrary number of membership functions per input feature.
    """
    
    def __init__(self, mf_distribution, outputs=1, membership_function=GeneralizedBell_MF(), output_type="default", features=None, dtype=torch.float32):
        """
        Initializes an ANFIS model.

        Args:
            mf_distribution (list[int]): List containing the number of membership functions per input feature.
            outputs (int): Number of model outputs. Defaults to ``1``.
            membership_function (MembershipFunction): Membership function to use. Defaults to ``GeneralizedBell_MF``.
            output_type (str): Output type of the model. Defaults to ``'default'``.
            features (iterable): Iterable of strings containing the names of the input features considered by the model. 
                Must be of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type to use in the model. Defaults to ``torch.float32``.
        """
        super(ANFIS, self).__init__()
        # Input data info
        self._input_size = len(mf_distribution)
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(self._input_size)]
        if features != None and len(features) == self._input_size:
            self.features = features
        
        
        # ANFIS structure info
        self._mf_distribution = torch.tensor(mf_distribution)
        self._rules = self._mf_distribution.prod().item()
        self._mfs = self._mf_distribution.sum().item()
        
        
        # Output info
        self._output_type = output_type
        self._outputs = outputs
        self._classes = torch.arange(outputs)
        self._custom_classes = False
        
        
        # Layers
        self._fuzzification_layer = FuzzificationLayer(
            mf_distribution=mf_distribution, 
            membership_function=membership_function,
            features=features,
            dtype=dtype)
        
        self._firing_levels_layer = FiringLevelsLayer(
            mf_distribution=self._mf_distribution
            )
        
        self._normalization_layer = NormalizationLayer()
        
        self._consequent_layer = ConsequentLayer(
            input_size=self._input_size,
            rules=self._rules,
            outputs=outputs,
            features=features,
            dtype=dtype)
        
        self._output_layer = OutputLayer(output_type=output_type)
    
    # ----- Premises seters and getters -----
    def get_premises(self):
        """
        Returns the premise parameters of the model.

        Returns:
            list[torch.Tensor]: List of tensors containing the premise parameters
            associated with each input feature, so the list length equals
            ``input_size``. Each tensor has shape ``(num_mfs, mf_params)``, where
            ``num_mfs`` is the number of membership functions for the corresponding
            feature and ``mf_params`` is the number of parameters of the membership
            function used.
        """
        return super().get_premises()
    
    def set_premises(self, premises):
        """
        Sets the membership function parameters of the model's fuzzification layer.
        
        Args:
            premises (list[torch.Tensor]): List of tensors containing the premise
                parameters. Each tensor must have shape ``(num_mfs, mf_params)``,
                where ``num_mfs`` is the number of membership functions for the
                corresponding feature and ``mf_params`` is the number of parameters
                of the membership function used.
        """
        super().set_premises(premises)
            
    def get_premises_as_parameters_list(self):
        """
        Returns the premise parameters of the model as a list of parameters.
        This is useful for optimization algorithms (using PyTorch optimizers).
        
        Returns:
            nn.ParameterList: A ParameterList containing the premise parameters
            for each input feature.
        """
        return super().get_premises_as_parameters_list()


    # ---- ANFIS parameters info ----
    @property
    def num_mfs(self):
        """
        Returns the number of membership functions per input feature.
        
        Returns:
            torch.Tensor: Tensor containing the number of membership functions for each input feature.
        """
        super().num_mfs
    
    
    # ----- Load state dict -----
    def load_state_dict(self, state_dict):
        """
        Loads a model state dictionary.
        
        Args:
            state_dict (dict): Dictionary containing the model state.
        """
        prems = []
        for i in range(self._input_size):
            prems.append(state_dict['_fuzzification_layer._premises.' + str(i)])
        self.set_premises(prems)
        self.set_consequents(state_dict['_consequent_layer._consequents'])
        
        
    # ----- Rules -----
    def get_rules_structure(self):
        """
        Returns a combined summary of the premises and consequent parameters for
        each rule in the model.

        The resulting DataFrame organizes each rule as a row, with columns grouped
        first by premises (showing the membership function parameters of each input
        feature) and then by the consequent parameters of each model output.

        Returns:
            pandas.DataFrame: DataFrame with a MultiIndex column structure, where
            the top-level groups correspond to ``'premises'`` and
            ``'output i consequents'`` for each output ``i``, and rows correspond
            to individual rules.
        """
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
    Rule-reduced Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Implements a homogeneous ANFIS model where every input feature shares the same number of membership functions, 
    restricting each feature to the same number of linguistic variables. Unlike the standard ANFIS, the number of
    rules is reduced by avoiding the full combinatorial expansion when computing firing levels. Instead, only the 
    membership degrees at matching indices across features are multiplied together, yielding a number of rules equal
    to the number of membership functions per feature. For further details, see :ref:`rule-reduced ANFIS <rule-reduced ANFIS>`.

    Note:
        Includes an experimental ``default_rule`` parameter that appends an extra firing level to capture input 
        combinations not covered by the reduced rule set. This functionality is not fully supported across all
        toolbox operations and is subject to change in future versions.

    Warning:
        The use of ``default_rule=True`` is experimental, is not supported in all toolbox operations, and may 
        produce unexpected behavior.
    """
    
    def __init__(self, input_size, num_mfs, outputs=1, default_rule=False, membership_function=GeneralizedBell_MF(), output_type="default", features=None, dtype=torch.float32):
        """
        Initializes a rule-reduced ANFIS model.

        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of membership functions per input feature.
            outputs (int): Number of model outputs. Defaults to ``1``.
            default_rule (bool): If ``True``, appends an extra firing level representing a default rule to capture 
                input combinations not covered by the reduced rule set. Defaults to ``False``.
            membership_function (MembershipFunction): Membership function to use. Defaults to ``GeneralizedBell_MF``.
            output_type (str): Output type of the model. Defaults to ``'default'``.
            features (iterable): Iterable of strings containing the names of the input features considered by the model. 
                Must be of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type to use in the model. Defaults to ``torch.float32``.

        .. caution:
            The ``default_rule`` parameter is under active development. Its
            behavior may change and some functionalities may not yet be available.
        """
        super(rule_reduced_ANFIS, self).__init__()
        self._rule_reduced = True
        self._default_rule = default_rule
        
        
        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
            
        
        # Output info
        self._output_type = output_type
        self._outputs = outputs
        self._classes = torch.arange(outputs)
        self._custom_classes = False
        
        
        # Layers
        self._fuzzification_layer = rule_reduced_FuzzificationLayer(
            input_size=input_size,
            num_mfs=num_mfs,
            membership_function=membership_function,
            features=features,
            dtype=dtype
            )
        
        self._firing_levels_layer = rule_reduced_FiringLevelsLayer(default_rule=default_rule)
        
        self._normalization_layer = NormalizationLayer(default_rule=default_rule)
        
        self._consequent_layer = alt_ConsequentLayer(
            input_size=input_size,
            rules=num_mfs,
            outputs=outputs,
            features=features,
            dtype=dtype
            )
        
        self._output_layer = OutputLayer(output_type=self._output_type)
        
        
    # ----- Premises seters and getters -----
    def get_premises_as_parameters_list(self):
        """
        Returns the premise parameters of the model as a list of parameters.
        This is useful for optimization algorithms.
        
        Returns:
            nn.ParameterList: A ParameterList containing the premise parameters of the membership functions.
        """
        return super().get_premises_as_parameters_list()
    
    
    # ----- Load state dict -----
    def load_state_dict(self, state_dict):
        """
        Loads a model state dictionary.
        
        Args:
            state_dict (dict): Dictionary containing the model state.
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
        """
        Returns a combined summary of the premises and consequent parameters for each rule in the model.

        The resulting DataFrame organizes each rule as a row, with columns grouped first by premises 
        (showing the membership function parameters of each input feature) and then by the consequent 
        parameters of each model output. Since this is a rule-reduced model, the number of rows equals 
        the number of membership functions per feature.

        Returns:
            pandas.DataFrame: DataFrame with a MultiIndex column structure, where the top-level groups correspond 
            to ``'premises'`` and ``'output i consequents'`` for each output ``i``, and rows correspond to individual rules.
        """
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
    
    
    # ---- rule reduced ANFIS rules operations ----
    def add_rules(self, means, stds):
        """
        Adds new rules to the rule-reduced ANFIS model by generating new membership
        functions from the provided means and standard deviations. The consequent
        parameters for the new rules are initialized randomly.
        
        Note:
            This method is agnostic to the membership function type used. Each
            membership function has a specific transformation function to convert
            the provided means and standard deviations into the corresponding
            membership function parameters.
        
        Args:
            means (torch.Tensor): Tensor containing the means for generating the
                new rules, of shape ``(num_new_rules, input_size)``, where
                ``num_new_rules`` is the number of rules to add and ``input_size``
                is the number of input features of the model.
            stds (torch.Tensor): Tensor containing the standard deviations for
                generating the new rules, of shape ``(num_new_rules, input_size)``,
                where ``num_new_rules`` is the number of rules to add and
                ``input_size`` is the number of input features of the model.
        """
        new_premises = self._fuzzification_layer._membership_function._grow_new_premise_parameters(means, stds)
        
        n_new_rules = new_premises.shape[1]
        
        new_consequents = self._consequent_layer._consequent_function.random_consequents(self._outputs, n_new_rules, self._input_size, self._dtype)
        
        self.set_premises(torch.cat((self.get_premises(), new_premises), dim=1))
        self.set_consequents(torch.cat((self.get_consequents(), new_consequents), dim=1))
        
    def remove_rules(self, rules_idxs):
        """
        Removes rules from the rule-reduced ANFIS model at the specified indices.
    
        Args:
            rules_idxs (list[int]): List of indices of the rules to remove. Each
                index must be an integer between ``0`` and ``num_rules - 1``, where
                ``num_rules`` is the current number of rules in the model.
        """
        premises = self.get_premises()
        consequents = self.get_consequents()
        
        mask = torch.ones(self.rules, dtype=torch.bool)
        mask[rules_idxs] = False
        
        new_premises = premises[:, mask, :]
        new_consequents = consequents[:, mask, :]
        
        self.set_premises(new_premises)
        self.set_consequents(new_consequents)