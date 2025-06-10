import torch

from neuro_fuzzy_toolbox.training import (
    base_model_trainer
)

import pandas as pd
import numpy as np

class SONFIS(base_model_trainer):
    """
    Self-organizing neuro-fuzzy inference system algorithm. Usa el algoritmo de aprendizaje híbrido ANFIS para actualizar los parámetros del modelo junto con las operaciones GrowNet, SplitSubNet y VanishNet para actualizar su estructura.
    Esta clase herea de la clase base_model_trainer para algunos métodos en común.
    
    Note:
        El modelo ANFIS al que se le aplique debe ser rule-reduced, por lo que será una instancia de la clase h_ANFIS.
    """
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, early_stopping=None, lse_for_new_consequents=False, lse_for_new_consequents_lambda=0., last_training_iteration=False):
        """
        Inicializa una nueva instancia de SONFIS.
        
        Args:
            Ngrow (int): Número mínimo de muestras mal modeladas para crecer una nueva subred.
            dGrow (float): Si el nivel de disparo máximo de una muestra es menor o igual a este valor elevado a la dimensión de los datos de entrada, entonces dicha muestra se considera mal modelada.
            Nsplit (int): Número mínimo de muestras para dividir una subred.
            eSplit (float): Error cuadrado medio mínimo de un conjunto de muestras para dividir una subred.
            Nvanish (int): Número máximo de muestras para desvanecer una subred.
            lVanish (int): Edad máxima para desvanecer una subred.
            max_iterations (int): Número máximo de iteraciones.
            ANFIStrainer (ANFIS learning algorithm instance): Instancia de algún algoritmo de entrenamiento para los modelos ANFIS. Básicamente define la manera en la se actualizarán los parámetros del modelo.
            early_stopping (EarlyStopping): Instancia de EarlyStopping (Default: None).
            lse_for_new_consequents (bool): Indica si los parámetros consecuentes de las nuevas reglas generadas (por GrowNet o SplitSubNetwork) deben ser inicializados con estimación de mínimos cuadrados usando "tridiagonal reduction and SVD", en caso de ser False, se inicializarían aleatoriamente (Default: False).
            lse_for_new_consequents_lambda (float): Lambda usado para utilizar "Regularización Ridge" en la inicialización de consecuentes con mínimos cuadrados luego de generar las reglas correspondientes (por GrowNet o SplitSubNetwork). Si es 0, no se realiza regularización (Default: 0.).
            last_training_iteration (bool): Indica si se debe realizar una última actualización de parámetros después de que el algoritmo SONFIS finalice (Default: False).
        """
        # ------------- SONFIS -------------
        # Hyperparameters
        self.Ngrow = Ngrow
        self.dGrow = dGrow
        self.Nsplit = Nsplit
        self.eSplit = eSplit
        self.Nvanish = Nvanish
        self.lVanish = lVanish
        self.max_iterations = max_iterations
        self.lse_for_new_consequents = lse_for_new_consequents
        self.lse_for_new_consequents_lambda = lse_for_new_consequents_lambda
        self.last_training_iteration = last_training_iteration
        
        # Early stopping
        self.sonfis_early_stopping = early_stopping
        
        # history
        self.history = {"loss": []}
        self.val_history = {"loss": []}
        
        
        # --------- ANFIS trainer ---------
        self.trainer = ANFIStrainer
        self.loss_function = self.trainer.loss_function
        
        # Optimizer
        self._optimizer_instance = None
        
        
        # ------ Internal variables & methods ------
        self._freezed = torch.tensor([], dtype=torch.int)
        self._ages = torch.tensor([], dtype=torch.int)
        self._last_best_rules = torch.tensor([-1])
        
        self._get_max_firing_level = None
        
        
        # ------ Rules Dataframe ------
        self._rules_dataframe = None
        self._current_best_rules_dataframe = None
        self._current_max_idx = 0
        self._best_rules_dataframe_iter = None
    
    
    def __call__(self, ANFISmodel, train_loader, val_loader=None, verbose=True):
        """
        Ejecuta el algoritmo SONFIS.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            loader (DataLoader): Instancia de DataLoader.
            verbose (bool): Si es True, imprime el progreso del entrenamiento (Default: True).
            
        """
        if ANFISmodel._rule_reduced == False:
            raise ValueError('The ANFIS model must be rule reduced')
        self._set_max_firing_level_method(ANFISmodel)
        
        self._current_max_idx = ANFISmodel.rules - 1
            
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        self._ages = torch.zeros(ANFISmodel.rules, dtype=torch.int)
        self._freezed = torch.zeros(ANFISmodel.rules, dtype=torch.int).bool()
        
        iter_width = len(str(self.max_iterations))
        print(f'ITERATION: {0:{iter_width}}/{self.max_iterations}')
        
        self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
        
        if verbose:
            early_stop_flag = False
            self._rules_dataframe = ANFISmodel.get_rules_structure().reset_index(drop=True)
            if val_loader is not None:
                self._current_best_rules_dataframe = self._rules_dataframe.copy()
                self._best_rules_dataframe_iter = 0
            print("\nSTARTING STATE:")
            print(self._rules_dataframe.to_string())

        if val_loader is not None:
            print(f'\n\tloss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
        else:
            print(f'\n\tloss: {self.history["loss"][-1]:.6f}')
        print(f'\t --> ANFIS rules: {ANFISmodel.rules}\n')
        
        model_updated = True
        i = 0
        while(model_updated and i < self.max_iterations):
            
            print(f'\nITERATION: {i+1:{iter_width}}/{self.max_iterations}')
            
            self._freeze_subnets()
            
            model_updated = self._update_structure(ANFISmodel, train_loader, verbose)
            
            if model_updated:
                self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
                
                if verbose:
                    self._replace_trained_subnets_on_rules_dataframe(ANFISmodel)
                    print("\nCURRENT STATE:")
                    print(self._rules_dataframe.to_string())
                    
                self._register_loss(ANFISmodel, train_loader, val_loader)
                
                if val_loader is not None:
                    print(f'\n\tloss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
                else:
                    print(f'\n\tloss: {self.history["loss"][-1]:.6f}')
                print(f'\t --> ANFIS rules: {ANFISmodel.rules}\n')
            
                if (val_loader is not None) and (self.sonfis_early_stopping is not None):
                    if self._check_early_stop(ANFISmodel, self.val_history["loss"][-1]):
                        print(f'found on the {self._best_rules_dataframe_iter}° iteration.')
                        early_stop_flag = True
                        break
                    else:
                        if self.sonfis_early_stopping._counter == 0:
                            self._current_best_rules_dataframe = self._rules_dataframe.copy()
                            self._best_rules_dataframe_iter = i+1
            
            else:
                print('NO MORE UPDATES')
            
            i += 1
            
        if i == self.max_iterations: print('MAX ITERATIONS REACHED')
            
        self._unfreeze_subnets()
        
        if self.last_training_iteration:
            print('\nLast training iteration (all subnets are being trained again)')
            
            self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
            
            if verbose:
                if early_stop_flag:
                    self._rules_dataframe = ANFISmodel.get_rules_structure().reset_index(drop=True)
                else:
                    self._replace_trained_subnets_on_rules_dataframe(ANFISmodel)
                    
                print("LAST TRAINING STATE:")
                print(self._rules_dataframe.to_string())
            
            self._register_loss(ANFISmodel, train_loader, val_loader)

            if val_loader is not None:
                print(f'\tloss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
            else:
                print(f'\tloss: {self.history["loss"][-1]:.6f}')
            
        print('\nTRAINING FINISHED')
        print(f'\t --> ANFIS rules: {ANFISmodel.rules}\n')
        if verbose:
            if early_stop_flag:
                self._rules_dataframe = ANFISmodel.get_rules_structure().reset_index(drop=True)
        print(self._rules_dataframe.to_string())
        
    
    # ----- Freezed subnets -----
    def _freeze_subnets(self):
        """
        Congela todas las subredes.
        """
        self._freezed = torch.ones_like(self._freezed).bool()
        
    def _unfreeze_subnets(self):
        """
        Descongela todas las subredes.
        """
        self._freezed = torch.zeros_like(self._freezed).bool()
    
    
    # ----- Early Stopping -----
    def _check_early_stop(self, ANFISmodel, loss):
        """
        Chequea si se cumple la condición de la instancia de EarlyStopping (si existe).
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            loss (float): Valor actual de la pérdida.
        
        Returns:
            bool: Indica si se debe detener el entrenamiento.
        """
        if self.sonfis_early_stopping is not None:
            self.sonfis_early_stopping(ANFISmodel, loss, verbose=True)
            if self.sonfis_early_stopping.stop:
                self._ages = torch.zeros(ANFISmodel.rules, dtype=torch.int)
                self._freezed = torch.zeros(ANFISmodel.rules, dtype=torch.int).bool()
                self.sonfis_early_stopping.reset()
                return True
        return False
    
    
    # ----- Rules dataframe -----
    def _add_subnets_on_rules_dataframe(self, new_premises, new_consequents):
        n_new_subnets = new_premises.shape[1]
        
        new_premises = new_premises.permute(1, 0, 2).reshape(n_new_subnets, -1)
        new_consequents = new_consequents.permute(1, 0, 2).reshape(n_new_subnets, -1)
        
        data_block = torch.cat([new_premises, new_consequents], dim=1).cpu().numpy()
        
        new_subnets_df = pd.DataFrame(data_block, columns=self._rules_dataframe.columns)
                
        start = self._current_max_idx + 1
        new_subnets_df.index = [i for i in range(start, start + n_new_subnets)]
        self._current_max_idx = new_subnets_df.index[-1]
        
        if self._rules_dataframe.empty:
            self._rules_dataframe = new_subnets_df
        else:
            self._rules_dataframe = pd.concat([self._rules_dataframe, new_subnets_df], axis=0)
        
    def _drop_subnets_on_rules_dataframe(self, mask):
        keep_mask = ~mask.cpu().numpy()
        
        self._rules_dataframe = self._rules_dataframe.loc[keep_mask]
        
    def _drop_subnets_on_rules_dataframe_by_idxs(self, idxs_tensor):
        idxs = idxs_tensor.cpu().numpy()
        keep = np.ones(len(self._rules_dataframe), dtype=bool)
        keep[idxs] = False
        
        self._rules_dataframe = self._rules_dataframe.iloc[keep]
        
    def _replace_trained_subnets_on_rules_dataframe(self, ANFISmodel):
        mask = ~self._freezed
        mask_np = mask.cpu().numpy().astype(bool)
        
        new_premises = ANFISmodel.get_premises()[:, mask, :]
        new_consequents = ANFISmodel.get_consequents()[:, mask, :]
        
        n_new_subnets = new_premises.shape[1]
        if n_new_subnets == 0: return
        
        new_premises = new_premises.permute(1, 0, 2).reshape(n_new_subnets, -1)
        new_consequents = new_consequents.permute(1, 0, 2).reshape(n_new_subnets, -1)
        
        data_block = torch.cat([new_premises, new_consequents], dim=1).cpu().numpy()
        
        self._rules_dataframe.iloc[mask_np, :] = data_block
        
    
    # ----- Internal Methods -----
    def _set_max_firing_level_method(self, ANFISmodel):
        """
        Asigna un método para obtener el máximo nivel de disparo de cada sample dentro de un batch dependiendo si el modelo ANFIS tiene (o no) regla por defecto.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            
        """
        if ANFISmodel._default_rule:
            self._get_max_firing_level = lambda firing_levels: torch.max(firing_levels[:, :-1], dim=1)
        else: 
            self._get_max_firing_level = lambda firing_levels: torch.max(firing_levels, dim=1)
        
    
    
    # ----- Update structure -----
    def _update_structure(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta las operaciones GrowNet, SplitSubNet y VanishNet para actualizar la estructura del modelo ANFIS.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.

        Returns:
            bool: Indica si se realizó alguna actualización en la estructura del modelo.
        """
        did_Grow = self._GrowNet(ANFISmodel, train_loader, verbose)
        
        did_Split = False
        if not did_Grow:
            did_Split = self._SplitSubNet(ANFISmodel, train_loader, verbose)
        
        did_Vanish = self._VanishNet(ANFISmodel, train_loader, verbose)

        return did_Grow or did_Split or did_Vanish
    
    
    def _GrowNet(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta el operador GrowNet para generar nuevas subredes.
        
        Args:
            ANFISmodel (rule_reduced_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
        
        Returns:
            bool: Indica si se generó alguna nueva subred.
        """
        bad_samples = torch.tensor([])
        bad_targets = torch.tensor([])
        best_bs_rules = torch.tensor([], dtype=torch.int)
        for batch_x, batch_y in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = self._get_max_firing_level(firing_levels)
            
            dGrowMask = max_fl.values <= self.dGrow**ANFISmodel._input_size # using dGrow
            
            bad_samples = torch.cat((bad_samples, batch_x[dGrowMask]), dim=0) # collect bad samples based on its max firing levels
            bad_targets = torch.cat((bad_targets, batch_y[dGrowMask]), dim=0)
            best_bs_rules = torch.cat((best_bs_rules, max_fl.indices[dGrowMask]), dim=0) # & the associated subnet
            
        unique_rules, counts = torch.unique(best_bs_rules, return_counts=True) # how many "max firing levels" do each of the subnets get?
        Ngrow_mask = counts > self.Ngrow
        
        indices_to_keep = torch.isin(best_bs_rules, unique_rules[Ngrow_mask]).nonzero().squeeze()  # using Ngrow
        
        bad_samples = bad_samples[indices_to_keep] # getting which samples will be considered
        bad_targets = bad_targets[indices_to_keep]
        best_bs_rules = best_bs_rules[indices_to_keep] # & the associated subnet
        
        if bad_samples.size(0) == 0:
            return False
        
        else:
            rules = [best_bs_rules == rule for rule in torch.unique(best_bs_rules)] # list of boolean masks (lenght: current number of subnets), each one with shape: (bad_samples.shape[0], ) 
            
            means = torch.stack([bad_samples[rule].mean(dim=0) for rule in rules]) # shape = (new_subnets, input_dim)
            stds = torch.stack([bad_samples[rule].std(dim=0) for rule in rules]) # shape = (new_subnets, input_dim)
            
            new_premises = ANFISmodel._fuzzification_layer._membership_function._grow_new_premise_parameters(means, stds)
            ANFISmodel.set_premises(torch.cat((ANFISmodel.get_premises(), new_premises), dim=1))
            
            n_new_rules = new_premises.shape[1]
            new_consequents = ANFISmodel._consequent_layer._consequent_function.random_consequents(ANFISmodel._outputs, n_new_rules, ANFISmodel._input_size, ANFISmodel._dtype)
            ANFISmodel.set_consequents(torch.cat((ANFISmodel.get_consequents(), new_consequents), dim=1))
            
            if self.lse_for_new_consequents: # if True, consequents are init with LSE
                new_consequents = self._lse_after_GrowNet(ANFISmodel, bad_samples, bad_targets, rules, n_new_rules)
                ANFISmodel.set_consequents(torch.cat((ANFISmodel.get_consequents()[:, :-n_new_rules, :], new_consequents), dim=1))
            
            if verbose:
                self._add_subnets_on_rules_dataframe(new_premises, new_consequents)
                print(f"\t-> Growing {n_new_rules} new subnets: {[i for i in range(self._current_max_idx - new_premises.shape[1] + 1, self._current_max_idx + 1)]}")
            
            self._ages = torch.cat((self._ages, torch.zeros(new_premises.shape[1], dtype=torch.int)))
            self._freezed = torch.cat((self._freezed, torch.zeros(new_premises.shape[1], dtype=torch.int).bool()))
            
            return True
        
    def _lse_after_GrowNet(self, ANFISmodel, samples, targets, rules_mask, n_new_rules):
        """_summary_

        Args:
            ANFISmodel (rule_reduced_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            samples (torch.Tensor): Datos de entrada que se consideran para la creación de las nuevas reglas (o subredes)
            targets (torch.Tensor): Targets asociados a los samples considerados para la creación de las nuevas reglas (o suredes).
            rules_mask (list): Lista de largo "n_new_rules" que contiene máscaras, las cuales indican a qué reglas se asocian cada uno de los elementos de los tensores "samples" y "targets".
            n_new_rules (int): Cantidad de nuevas reglas (o subredes) que se generaron en el método GrowNet.

        Returns:
            torch.Tensor: Parámetros consecuentes estimados con mínimos cuadrados de las reglas agregadas en la operación GrowNet
        """
        new_consequents = torch.tensor([])
        
        i = 0
        for rule in rules_mask:
            x = samples[rule]
            y = targets[rule]
            
            _, w_norm, _ = ANFISmodel.intermediate_values(x)
            xe = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
            fs = w_norm[:, i - n_new_rules].unsqueeze(0).t()
            
            '''preliminary fix for the dtype issue'''
            if ANFISmodel._output_type == 'softmax':
                y = y.to(torch.int64)
                y = torch.nn.functional.one_hot(y, ANFISmodel._outputs)
            if y.dtype != xe.dtype:
                y = y.to(xe.dtype)
            '''preliminary fix for the dtype issue'''
            
            A = xe * fs
            
            if self.lse_for_new_consequents_lambda > 0.:
                p = A.shape[1]
                I = torch.eye(p, dtype=A.dtype) * torch.sqrt(torch.tensor(self.lse_for_new_consequents_lambda, dtype=A.dtype))
                A = torch.cat([A, I], dim=0)
                if y.dim() > 1:
                    m = y.shape[1]
                    zeros = torch.zeros((p, m), dtype=A.dtype)
                else:
                    zeros = torch.zeros(p, dtype=A.dtype)
                y  = torch.cat([y, zeros], dim=0)
            
            C, _, _, _ = torch.linalg.lstsq(A, y, rcond=None, driver="gelsd")
            new_consequents = torch.cat((new_consequents, C.t().reshape(ANFISmodel._outputs, 1, xe.shape[1])), dim=1)
            
            i += 1
        
        return new_consequents
    
    def _SplitSubNet(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta el operador SplitSubNetwork para dividir subredes.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
            
        Returns:
            bool: Indica si se dividió alguna subred.
        """
        samples = torch.tensor([])
        targets = torch.tensor([])
        best_rules = torch.tensor([], dtype=torch.int)

        for batch_x, batch_y in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = self._get_max_firing_level(firing_levels)
            
            samples = torch.cat((samples, batch_x), dim=0) # collect the samples,
            targets = torch.cat((targets, batch_y), dim=0) # the associated targets
            best_rules = torch.cat((best_rules, max_fl.indices), dim=0) # & the associated subnet (with the max firing level)
            
        with torch.no_grad():
            model_outputs = ANFISmodel(samples, return_probs=False) # get model preds
            
        unique_rules, counts = torch.unique(best_rules, return_counts=True) # how many "max firing levels" do each of the subnets get?
        Nsplit_mask = counts > self.Nsplit
        
        indices_to_keep = torch.isin(best_rules, unique_rules[Nsplit_mask]).nonzero().squeeze()  # using Nsplit
        
        if indices_to_keep.size(0) == 0:
            return False
        
        else:
            # collect the samples, targets, outputs and the "best rule associated" (based on the max firing level) to be considered
            model_outputs = model_outputs[indices_to_keep]
            samples = samples[indices_to_keep]
            targets = targets[indices_to_keep]
            best_rules = best_rules[indices_to_keep]
            
            unique_rules = torch.unique(best_rules)
            
            if targets.dtype != train_loader.dataset.tensors[1].dtype:
                targets = targets.to(train_loader.dataset.tensors[1].dtype)
                
            rules = [best_rules == rule for rule in unique_rules] # list of boolean masks (lenght: current number of subnets), each one with shape: (bad_samples.shape[0], ) 
            
            model_outputs_list = [model_outputs[rule] for rule in rules]
            targets_list = [targets[rule] for rule in rules]
                
            # compute loss 
            loss_values = torch.stack([self.loss_function(model_outputs_list[i], targets_list[i]) for i in range(len(rules))]) # for each of the considered subnets with ONLY its associated samples
            
            eSplit_mask = loss_values > self.eSplit
            
            rules_to_split = unique_rules[eSplit_mask]
            
            if ((targets.shape[0] == 0) or (rules_to_split.shape[0] == 0)): # using eSplit
                return False
            
            else:
                if self.lse_for_new_consequents:
                    to_split_samples_list, to_split_targets_list = self._group_samples_for_lse_in_order(eSplit_mask, targets_list, samples, rules)
                    
                new_premises = ANFISmodel.get_premises()
                new_consequents = ANFISmodel.get_consequents()
                
                all_new_premises = torch.tensor([])
                all_new_consequents = torch.tensor([])
                
                idx = 0
                for rule in list(torch.flip(rules_to_split, [0]).long()): # using eSplit
                    new_premises = torch.cat((new_premises[:, :rule,:], new_premises[:, rule+1:, :]), dim=1)
                    to_split = ANFISmodel.get_premises()[:, rule:rule+1, :]
                    split = ANFISmodel._fuzzification_layer._membership_function._split_premise_parameters(to_split)
                    
                    new_premises = torch.cat((new_premises, split), dim=1)
                    
                    new_consequents = torch.cat((new_consequents[:, :rule, :], new_consequents[:, rule+1:, :]), dim=1)
                    new_consequent_to_add = ANFISmodel._consequent_layer._consequent_function.random_consequents(ANFISmodel._outputs, 2, ANFISmodel._input_size, ANFISmodel._dtype)
                    
                    new_consequents = torch.cat((new_consequents, new_consequent_to_add), dim=1)
                    
                    self._ages = torch.cat((self._ages[:rule], self._ages[rule+1:]))
                    self._freezed = torch.cat((self._freezed[:rule], self._freezed[rule+1:]))
                    self._ages = torch.cat((self._ages, torch.zeros(2, dtype=torch.int)))
                    self._freezed = torch.cat((self._freezed, torch.zeros(2, dtype=torch.int).bool()))
                    
                    if verbose:
                        all_new_premises = torch.cat((all_new_premises, split), dim=1)
                        all_new_consequents = torch.cat((all_new_consequents, new_consequent_to_add), dim=1)
                    
                    ANFISmodel.set_premises(new_premises)
                    ANFISmodel.set_consequents(new_consequents)
                    
                    if self.lse_for_new_consequents:
                        new_2_last_consequents_to_replace = self._lse_while_SplitSubNet(ANFISmodel, to_split_samples_list[idx], to_split_targets_list[idx])
                        ANFISmodel.set_consequents(torch.cat((ANFISmodel.get_consequents()[:, :-2, :], new_2_last_consequents_to_replace), dim=1))
                        
                    new_premises = ANFISmodel.get_premises()
                    new_consequents = ANFISmodel.get_consequents()
                        
                    idx += 1
                
                if verbose:
                    subnets = rules_to_split.tolist()
                    if rules_to_split[rules_to_split == True].size(0) == 1:
                        subnets = [subnets]
                    #if isinstance(self._rules_dataframe.index[subnets[0]], int):
                    #    #print(f"\t-> self._rules_dataframe.index[i]: {self._rules_dataframe.index[subnets[0]]}")
                    #    print(f"\t-> Splitting {rules_to_split.shape[subnets[0]]} subnets: {[self._rules_dataframe.index[i] for i in subnets]}")
                    #else:
                    #    print(f"\t-> Splitting {rules_to_split.shape[0]} subnets: {[self._rules_dataframe.index[i].item() for i in subnets]}")
                    print(f"\t-> Splitting {rules_to_split.shape[0]} subnets: {subnets}")
                    
                    self._drop_subnets_on_rules_dataframe_by_idxs(rules_to_split)
                    self._add_subnets_on_rules_dataframe(all_new_premises, all_new_consequents)
                
                return True
            
    def _group_samples_for_lse_in_order(self, eSplit_mask, targets_list, samples, rules):
        to_split_samples_list = []
        to_split_targets_list = []
        
        i = eSplit_mask.shape[0] - 1
        for boolean in torch.flip(eSplit_mask, [0]):
            if boolean:
                to_split_targets_list.append(targets_list[i])
                to_split_samples_list.append(samples[rules[i]])
            i -= 1
            
        return to_split_samples_list, to_split_targets_list
            
    def _lse_while_SplitSubNet(self, ANFISmodel, samples, targets):
        new_consequents = torch.tensor([])
        
        x = samples
        y = targets
        
        _, w_norm, _ = ANFISmodel.intermediate_values(x)
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        w_norm = w_norm[:, -2:].unsqueeze(2).repeat(1, 1, x.shape[1]).view(w_norm[:, -2:].shape[0], -1)
        X = x.repeat(1, 2)
        
        '''preliminary fix for the dtype issue'''
        if ANFISmodel._output_type == 'softmax':
            y = y.to(torch.int64)
            y = torch.nn.functional.one_hot(y, ANFISmodel._outputs)
        if y.dtype != X.dtype:
            y = y.to(X.dtype)
        '''preliminary fix for the dtype issue'''
        
        A = X * w_norm
        
        if self.lse_for_new_consequents_lambda > 0.:
            p = A.shape[1]
            I = torch.eye(p, dtype=A.dtype) * torch.sqrt(torch.tensor(self.lse_for_new_consequents_lambda, dtype=A.dtype))
            A = torch.cat([A, I], dim=0)
            if y.dim() > 1:
                m = y.shape[1]
                zeros = torch.zeros((p, m), dtype=A.dtype)
            else:
                zeros = torch.zeros(p, dtype=A.dtype)
            y  = torch.cat([y, zeros], dim=0)
        
        C, _, _, _ = torch.linalg.lstsq(A, y, rcond=None, driver="gelsd")
        new_consequents = C.t().reshape(ANFISmodel._outputs, 2, x.shape[1])
        
        return new_consequents
    
    def _VanishNet(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta el operador VanishNet para desvanecer subredes.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
            
        Returns:
            bool: Indica si se desvaneció alguna subred.
        """
        best_rules = torch.tensor([], dtype=torch.int)
        
        for batch_x, _ in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = self._get_max_firing_level(firing_levels) 
            
            best_rules = torch.cat((best_rules, max_fl.indices), dim=0) # collect only the subnet with the max firing level for each sample
            
        unique_rules, counts = torch.unique(best_rules, return_counts=True) # how many "max firing levels" do each of the subnets get?
        all_rules = torch.arange(ANFISmodel.rules)
        
        total_counts = torch.zeros(ANFISmodel.rules, dtype=torch.int64)
        total_counts[unique_rules] = counts
        
        if torch.equal(best_rules, self._last_best_rules) or torch.equal(self._last_best_rules, torch.tensor([-1])): # if the "best_rule" for each sample is the same
            self._ages += 1 # add 1 age to all subnets
        else: # if the "best_rule" for each sample is different in this SONFIS iteration --> # only add 1 age to the subnets that NOT improved (have less "associated" samples than before)
            last_unique_rules, last_counts = torch.unique(self._last_best_rules, return_counts=True)
            self._last_best_rules = best_rules
            
            last_total_counts = torch.zeros(ANFISmodel.rules, dtype=torch.int64)
            last_total_counts[last_unique_rules[last_unique_rules < ANFISmodel.rules]] = last_counts[last_unique_rules < ANFISmodel.rules]
            
            improved_rules = all_rules[(total_counts < last_total_counts)]
            
            self._ages[improved_rules] = 0
            self._ages[~improved_rules] += 1
        
        mask = ((self._ages > self.lVanish) & (total_counts < self.Nvanish)) # using Nvanish & lVanish --> to filter by age 6 by number of "associated" samples
        rules_to_eliminate = all_rules[mask]
        
        if torch.equal(rules_to_eliminate, torch.tensor([], dtype=torch.int64)): # if there ARE NOT rules to eliminate
            return False
        else: # if there ARE rules to eliminate
            new_premises = ANFISmodel.get_premises()
            new_consequents = ANFISmodel.get_consequents()
            for rule in torch.flip(rules_to_eliminate, dims=(0,)):
                new_premises = torch.cat((new_premises[:, :rule, :], new_premises[:, rule+1:, :]), dim=1)
                new_consequents = torch.cat((new_consequents[:, :rule, :], new_consequents[:, rule+1:, :]), dim=1)
                
                self._ages = torch.cat((self._ages[:rule], self._ages[rule+1:]))
                self._freezed = torch.cat((self._freezed[:rule], self._freezed[rule+1:]))
            
            ANFISmodel.set_premises(new_premises)
            ANFISmodel.set_consequents(new_consequents)
            
            if verbose:
                subnets = (mask.nonzero().squeeze()).tolist()
                if mask[mask == True].size(0) == 1:
                    subnets = [subnets]
                print(f"\t-> Vanishing {rules_to_eliminate.size(0)} subnets: {[self._rules_dataframe.index[i].item() for i in subnets]}")
                self._drop_subnets_on_rules_dataframe(mask)
            
            return True