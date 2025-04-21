import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split

from EXP_neuro_fuzzy_toolbox.training import (
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
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation=0, early_stopping=None, last_training_iteration=False):
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
            validation (float): Porcentaje de los datos destinados a validación. Si es 0, no se realiza validación (Default: 0).
            early_stopping (EarlyStopping): Instancia de EarlyStopping (Default: None).
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
        self.last_training_iteration = last_training_iteration
        
        # Early stopping
        self.validation = validation
        self.sonfis_early_stopping = early_stopping
        
        # history
        self.history = {"loss": []}
        self.val_history = {"loss": []}
        
        
        # --------- ANFIS trainer ---------
        self.trainer = ANFIStrainer
        self.loss_function = self.trainer.loss_function
        self.trainer.validation = self.validation
        
        # Optimizer
        self._optimizer_instance = None
        
        
        # ------ Internal variables ------
        self._freezed = torch.tensor([], dtype=torch.int)
        self._ages = torch.tensor([], dtype=torch.int)
        self._last_best_rules = torch.tensor([-1])
        
        
        # ------ Rules Dataframe ------
        self._rules_dataframe = None
    
    
    def __call__(self, ANFISmodel, loader, verbose=True):
        """
        Ejecuta el algoritmo SONFIS.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            loader (DataLoader): Instancia de DataLoader.
            verbose (bool): Si es True, imprime el progreso del entrenamiento (Default: True).
            
        """
        if ANFISmodel._rule_reduced == False:
            raise ValueError('The ANFIS model must be rule reduced')
            
        train_loader, val_loader = self._train_val_split(ANFISmodel, loader)
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        self._ages = torch.zeros(ANFISmodel.rules, dtype=torch.int)
        self._freezed = torch.zeros(ANFISmodel.rules, dtype=torch.int).bool()
        
        self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
        
        if verbose:
            self._rules_dataframe = ANFISmodel.get_rules_structure()
            print("STARTING STATE:")
            print(self._rules_dataframe)

        iter_width = len(str(self.max_iterations))
        print("\n")
        if self.validation > 0:
            print(f'ITERATION: {0:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
        else:
            print(f'ITERATION: {0:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f}')
        print(f' --> ANFIS rules: {ANFISmodel.rules}\n')
        
        model_updated = True
        i = 0
        while(model_updated and i < self.max_iterations):
            
            self._freeze_subnets()
            
            model_updated = self._update_structure(ANFISmodel, train_loader, verbose)
            
            if model_updated:
                self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
                
                if verbose:
                    self._replace_trained_subnets_on_rules_dataframe(ANFISmodel)
                    print("\n\t-> Unfreezed subnets trained:")
                    print(self._rules_dataframe)

            else:
                print('NO MORE UPDATES')
                    
            self._register_loss(ANFISmodel, train_loader, val_loader)
            
            if self.validation > 0 and self._check_early_stop(ANFISmodel, self.val_history["loss"][-1]):
                break
            
            iter_width = len(str(self.max_iterations))
            print("\n")
            if self.validation > 0:
                print(f'ITERATION: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
            else:
                print(f'ITERATION: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f}')
            print(f' --> ANFIS rules: {ANFISmodel.rules}\n')
            
            i += 1
            
        if i == self.max_iterations: print('MAX ITERATIONS REACHED')
            
        self._unfreeze_subnets()
        
        if self.last_training_iteration:
            self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
            
            if verbose:
                self._replace_trained_subnets_on_rules_dataframe(ANFISmodel)
            
            self._register_loss(ANFISmodel, train_loader, val_loader)

            print('\nLast training iteration (all subnets were trained again)')
            if self.validation > 0:
                print(f'ITERATION: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
            else:
                print(f'ITERATION: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f}')
            
        print('\nTRAINING FINISHED')
        print(f' --> ANFIS rules: {ANFISmodel.rules}\n')
        
    
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
                return True
        return False
    
    
    # ----- Rules dataframe -----
    def _add_subnets_on_rules_dataframe(self, new_premises, new_consequents):
        n_new_subnets = new_premises.shape[1]
        
        new_premises = new_premises.permute(1, 0, 2).reshape(n_new_subnets, -1)
        new_consequents = new_consequents.permute(1, 0, 2).reshape(n_new_subnets, -1)
        
        data_block = torch.cat([new_premises, new_consequents], dim=1).cpu().numpy()
        
        new_subnets_df = pd.DataFrame(data_block, columns=self._rules_dataframe.columns)
        
        start = len(self._rules_dataframe) + 1
        new_subnets_df.index = [f"rule {i}" for i in range(start, start + n_new_subnets)]
        
        if self._rules_dataframe.empty:
            self._rules_dataframe = new_subnets_df
        else:
            self._rules_dataframe = pd.concat([self._rules_dataframe, new_subnets_df], axis=0)
        
    def _drop_subnets_on_rules_dataframe(self, mask):
        keep_mask = ~mask.cpu().numpy()
        
        self._rules_dataframe = self._rules_dataframe.loc[keep_mask]
        self._rules_dataframe.index = [f"rule {i+1}" for i in range(len(self._rules_dataframe))]
        
    def _drop_subnets_on_rules_dataframe_by_idxs(self, idxs_tensor):
        idxs = idxs_tensor.cpu().numpy()
        keep = np.ones(len(self._rules_dataframe), dtype=bool)
        keep[idxs] = False
        
        self._rules_dataframe = self._rules_dataframe.iloc[keep]
        self._rules_dataframe.index = [f"rule {i+1}" for i in range(len(self._rules_dataframe))]
        
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
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
        
        Returns:
            bool: Indica si se generó alguna nueva subred.
        """
        bad_samples = torch.tensor([])
        best_bs_rules = torch.tensor([], dtype=torch.int)
        for batch_x, _ in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            
            dGrowMask = max_fl.values <= self.dGrow**ANFISmodel._input_size
            
            bad_samples = torch.cat((bad_samples, batch_x[dGrowMask]), dim=0)
            best_bs_rules = torch.cat((best_bs_rules, max_fl.indices[dGrowMask]), dim=0)
            
        unique_rules, counts = torch.unique(best_bs_rules, return_counts=True)
        Ngrow_mask = counts > self.Ngrow
        
        indices_to_keep = torch.isin(best_bs_rules, unique_rules[Ngrow_mask]).nonzero().squeeze()
        
        bad_samples = bad_samples[indices_to_keep]
        best_bs_rules = best_bs_rules[indices_to_keep]
        
        if bad_samples.size(0) == 0:
            return False
        
        else:
            rules = [best_bs_rules == rule for rule in torch.unique(best_bs_rules)]
            
            means = torch.stack([bad_samples[rule].mean(dim=0) for rule in rules])
            stds = torch.stack([bad_samples[rule].std(dim=0) for rule in rules])
            
            new_premises = ANFISmodel._fuzzification_layer._membership_function._grow_new_premise_parameters(means, stds)
            ANFISmodel.set_premises(torch.cat((ANFISmodel.get_premises(), new_premises), dim=1))
            
            n_new_consequents = new_premises.shape[1]
            new_consequents = ANFISmodel._consequent_layer._consequent_function.random_consequents(ANFISmodel._outputs, n_new_consequents, ANFISmodel._input_size, ANFISmodel._dtype)
            ANFISmodel.set_consequents(torch.cat((ANFISmodel.get_consequents(), new_consequents), dim=1))
            
            if verbose:
                self._add_subnets_on_rules_dataframe(new_premises, new_consequents)
                print(f"\t-> Growing {n_new_consequents} new subnets:")
                print(self._rules_dataframe)
            
            self._ages = torch.cat((self._ages, torch.zeros(new_premises.shape[1], dtype=torch.int)))
            self._freezed = torch.cat((self._freezed, torch.zeros(new_premises.shape[1], dtype=torch.int).bool()))
            
            return True
            
    def _SplitSubNet(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta el operador SplitSubNetwork para dividir subredes.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
            
        Returns:
            bool: Indica si se dividió alguna subred.
        """
        targets = torch.tensor([])
        model_outputs = torch.tensor([])
        best_rules = torch.tensor([], dtype=torch.int)

        for batch_x, batch_y in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            
            with torch.no_grad():
                model_outputs = torch.cat((model_outputs, ANFISmodel(batch_x, return_probabilities=True)), dim=0)
            targets = torch.cat((targets, batch_y), dim=0)
            best_rules = torch.cat((best_rules, max_fl.indices), dim=0)
            
        unique_rules, counts = torch.unique(best_rules, return_counts=True)
        Nsplit_mask = counts > self.Nsplit
        
        indices_to_keep = torch.isin(best_rules, unique_rules[Nsplit_mask]).nonzero().squeeze()
        
        if indices_to_keep.size(0) == 0:
            return False
        
        else:
            model_outputs = model_outputs[indices_to_keep]
            targets = targets[indices_to_keep]
            best_rules = best_rules[indices_to_keep]
            
            unique_rules = torch.unique(best_rules)
            
            """
            premilinary fix for softmax output
            """
            if targets.shape != model_outputs.shape:
                targets = torch.nn.functional.one_hot(targets.to(torch.long), ANFISmodel._outputs)
            """
            premilinary fix for softmax output
            """
            mse_values = torch.stack([((targets[best_rules == rule] - model_outputs[best_rules == rule])**2).mean() for rule in unique_rules])
            
            eSplit_mask = mse_values > self.eSplit
            
            if ((targets.shape[0] == 0) or (unique_rules[eSplit_mask].shape[0] == 0)):
                return False
            
            else:
                new_premises = ANFISmodel.get_premises()
                new_consequents = ANFISmodel.get_consequents()
                
                all_new_premises = torch.tensor([])
                all_new_consequents = torch.tensor([])
                
                for rule in list(torch.flip(unique_rules[eSplit_mask], [0]).long()):
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
                
                if verbose:
                    self._drop_subnets_on_rules_dataframe_by_idxs(unique_rules[eSplit_mask])
                    self._add_subnets_on_rules_dataframe(all_new_premises, all_new_consequents)
                    print(f"\t-> Splitting {unique_rules[eSplit_mask].shape[0]} subnets: {(torch.flip(unique_rules[eSplit_mask], [0]).long()+1).tolist()}")
                    print(self._rules_dataframe)
                
                return True
    
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
            max_fl = torch.max(firing_levels, dim=1)
            
            best_rules = torch.cat((best_rules, max_fl.indices), dim=0)
            
        unique_rules, counts = torch.unique(best_rules, return_counts=True)
        all_rules = torch.arange(ANFISmodel.rules)
        
        total_counts = torch.zeros(ANFISmodel.rules, dtype=torch.int64)
        total_counts[unique_rules] = counts
        
        if torch.equal(best_rules, self._last_best_rules) or torch.equal(self._last_best_rules, torch.tensor([-1])):
            self._ages += 1
        else:
            last_unique_rules, last_counts = torch.unique(self._last_best_rules, return_counts=True)
            last_total_counts = torch.zeros(ANFISmodel.rules, dtype=torch.int64)
            last_total_counts[last_unique_rules[last_unique_rules < ANFISmodel.rules]] = last_counts[last_unique_rules < ANFISmodel.rules]
            
            improved_rules = all_rules[(total_counts < last_total_counts)]
            not_improved_rules = all_rules[(total_counts >= last_total_counts)]
            
            self._ages[improved_rules] = 0
            self._ages[not_improved_rules] += 1
        
        mask = ((self._ages > self.lVanish) & (total_counts < self.Nvanish))
        rules_to_eliminate = all_rules[mask]
        
        if torch.equal(rules_to_eliminate, torch.tensor([], dtype=torch.int64)):
            return False
        else:
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
                self._drop_subnets_on_rules_dataframe_by_idxs(rules_to_eliminate)
                print("\n")
                print(f"\t-> Vanishing {rules_to_eliminate.size(0)} subnets: {(torch.flip(rules_to_eliminate, dims=(0,))+1).tolist()}")
                print(self._rules_dataframe)
            
            self._last_best_rules = best_rules
            return True



class alt_SONFIS(SONFIS):
    """
    Self-organizing neuro-fuzzy inference system algorithm. Es una implementación alternativa de SONFIS que utiliza un enfoque diferente para las operaciones GrowNet, SplitSubNet y VanishNet.
    
    La única diferencia, además de la implementación de las operaciones, está en el operador VanishNet, que en este caso la edad de las subredes aumenta siempre que no alcancen a modelar Nvanish muestras y solo se reinicia si se supera dicho valor. (En el caso de SONFIS, la edad de una subred aumenta siempre que esta no mejore y se reinicia si lo hace).
    """
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation=0, early_stopping=None, last_training_iteration=False):
        """
        Inicializa una nueva instancia de alt_SONFIS.
        
        Args:
            Ngrow (int): Número mínimo de muestras para crecer una nueva subred.
            dGrow (float): Nivel mínimo de disparo para crecer una nueva subred.
            Nsplit (int): Número mínimo de muestras para dividir una subred.
            eSplit (float): Error mínimo para dividir una subred.
            Nvanish (int): Número mínimo de muestras para desvanecer una subred.
            lVanish (int): Edad máxima para desvanecer una subred.
            max_iterations (int): Número máximo de iteraciones.
            ANFIStrainer (Hybrid_learning_algorithm): Instancia del algoritmo de aprendizaje híbrido. Se utiliza para obtener sus parámetros, ya que el algoritmo de actualización de parámetros híbrido del modelo ANFIS clásico es por defecto el aplicado en SONFIS.
            validation (float): Porcentaje de división de validación. Si es 0, no se realiza validación (Default: 0).
            early_stopping (EarlyStopping): Instancia de EarlyStopping (Default: None).
            last_training_iteration (bool): Indica si se debe realizar una última actualización de parámetros después de que el algoritmo SONFIS finalice (Default: False).
        
        """
        super().__init__(Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation, early_stopping, last_training_iteration)
        
        
    def _GrowNet(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta el operador GrowNet para generar nuevas subredes.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
            
        Returns:
            bool: Indica si se generó alguna nueva subred.
        
        """
        x = torch.tensor([])
        w_max = torch.tensor([])
        related_rules = torch.tensor([], dtype=torch.bool)
        for batch_x, _ in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            w_max = torch.cat((w_max, max_fl.values))
            x = torch.cat((x, batch_x))
            related_rules = torch.cat((related_rules, torch.nn.functional.one_hot(max_fl.indices, ANFISmodel.rules).bool()))
        
        dGrowMask = w_max <= self.dGrow**ANFISmodel._input_size
        
        bad_x = x[dGrowMask]
        bad_related_rules = related_rules[dGrowMask]
        
        Ngrow_mask = bad_related_rules.sum(dim=0) > self.Ngrow
        
        samples_used_mask = bad_related_rules.t()[Ngrow_mask].t()

        counts = samples_used_mask.sum(dim=0, keepdim=True).t()
        reshaped_bad_x = bad_x.unsqueeze(1)
        
        means = (reshaped_bad_x * samples_used_mask.unsqueeze(-1)).sum(dim=0) / counts
        stds = (((((reshaped_bad_x - means.unsqueeze(0))**2) * samples_used_mask.unsqueeze(-1)).sum(dim=0)) / counts).sqrt()
        
        n_new_rules = means.size(0)
        
        if n_new_rules == 0:
            return False
        else:
            # Add new premises
            new_premises = ANFISmodel._fuzzification_layer._membership_function._grow_new_premise_parameters(means, stds)
            ANFISmodel.set_premises(torch.cat((ANFISmodel.get_premises(), new_premises), dim=1))
            
            # Add new consequents    
            new_consequents = ANFISmodel._consequent_layer._consequent_function.random_consequents(ANFISmodel._outputs, n_new_rules, ANFISmodel._input_size, ANFISmodel._dtype)
            ANFISmodel.set_consequents(torch.cat((ANFISmodel.get_consequents(), new_consequents), dim=1))
            
            # Update ages and freezed
            self._ages = torch.cat((self._ages, torch.zeros(n_new_rules, dtype=torch.int)))
            self._freezed = torch.cat((self._freezed, torch.zeros(n_new_rules, dtype=torch.int).bool()))
            
            if verbose:
                self._add_subnets_on_rules_dataframe(new_premises, new_consequents)
                print(f"\t-> Growing {new_premises.shape[1]} new subnets:")
                print(self._rules_dataframe)
            
            return True
        
        
    def _SplitSubNet(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta el operador SplitSubNetwork para dividir subredes.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
            
        Returns:
            bool: Indica si se dividió alguna subred.
        
        """
        y = torch.tensor([])
        related_rules = torch.tensor([], dtype=torch.bool)
        model_outputs = torch.tensor([])
        for batch_x, batch_y in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            y = torch.cat((y, batch_y))
            with torch.no_grad():
                model_outputs = torch.cat((model_outputs, ANFISmodel(batch_x, return_probabilities=True)))
            related_rules = torch.cat((related_rules, torch.nn.functional.one_hot(max_fl.indices, ANFISmodel.rules).bool()))
        
        """
        premilinary fix for softmax output
        """
        if y.shape != model_outputs.shape:
            y = torch.nn.functional.one_hot(y.to(torch.long), ANFISmodel._outputs)
        """
        premilinary fix for softmax output
        """
        
        expanded_related_rules = related_rules.t().unsqueeze(-1)
        
        masked_y = expanded_related_rules * y
        masked_outputs = expanded_related_rules * model_outputs
        counts = related_rules.sum(dim=0)
        mse_counts = counts*ANFISmodel._input_size
        
        mse_by_group = (((masked_y - masked_outputs) ** 2).sum(dim=1).sum(dim=1))/mse_counts
        
        NsplitMask = counts > self.Nsplit
        eSplitMask = mse_by_group > self.eSplit
        
        rules_to_split = NsplitMask*eSplitMask
        
        if rules_to_split[rules_to_split == True].size(0) == 0:
            return False
        
        else:
            self._ages = self._ages[~rules_to_split]
            self._freezed = self._freezed[~rules_to_split]
            
            # Split premises
            still_premises = ANFISmodel.get_premises()[:, ~rules_to_split, :]
            split_premises = ANFISmodel._fuzzification_layer._membership_function._split_premise_parameters(ANFISmodel.get_premises()[:, rules_to_split, :])
            new_premises = torch.cat((still_premises, split_premises), dim=1)
            ANFISmodel.set_premises(new_premises)
            
            # New Consequents
            still_consequents = ANFISmodel.get_consequents()[:, ~rules_to_split, :]
            split_consequents = ANFISmodel._consequent_layer._consequent_function.random_consequents(ANFISmodel._outputs, split_premises.shape[1], ANFISmodel._input_size, ANFISmodel._dtype)
            new_consequents = torch.cat((still_consequents, split_consequents), dim=1)
            ANFISmodel.set_consequents(new_consequents)

            # Update ages and freezed
            new_ages = torch.zeros(rules_to_split[rules_to_split == True].size(0) * 2, dtype=torch.int)
            self._ages = torch.cat((self._ages, new_ages))
            
            new_freezed = torch.zeros(rules_to_split[rules_to_split == True].size(0) * 2, dtype=torch.int).bool()
            self._freezed = torch.cat((self._freezed, new_freezed))
            
            if verbose:
                self._drop_subnets_on_rules_dataframe(rules_to_split)
                self._add_subnets_on_rules_dataframe(split_premises, split_consequents)
                print(f"\t-> Splitting {rules_to_split[rules_to_split == True].size(0)} subnets: {(rules_to_split.nonzero().squeeze()+1).tolist()}")
                print(self._rules_dataframe)
            
            return True
        
        
    def _VanishNet(self, ANFISmodel, train_loader, verbose):
        """
        Ejecuta el operador VanishNet para desvanecer subredes.
        
        Args:
            ANFISmodel (h_ANFIS): Instancia del modelo ANFIS reducido en reglas.
            train_loader (DataLoader): DataLoader de entrenamiento.
            
        Returns:
            bool: Indica si se desvaneció alguna subred.
        
        """
        related_rules = torch.tensor([], dtype=torch.bool)
        for batch_x, _ in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            related_rules = torch.cat((related_rules, torch.nn.functional.one_hot(max_fl.indices, ANFISmodel.rules).bool()))
        
        counts = related_rules.sum(dim=0)
        
        ages_update = (counts < self.Nvanish).int()
        self._ages *= ages_update
        self._ages += ages_update
        
        lVanishMask = self._ages > self.lVanish
        
        if lVanishMask.sum() == 0:
            return False
        
        else:
            new_premises = ANFISmodel.get_premises()[:, ~lVanishMask, :]
            new_consequents = ANFISmodel.get_consequents()[:, ~lVanishMask, :]
            
            self._ages = self._ages[~lVanishMask]
            self._freezed = self._freezed[~lVanishMask]
            
            ANFISmodel.set_premises(new_premises)
            ANFISmodel.set_consequents(new_consequents)
            
            if verbose:
                self._drop_subnets_on_rules_dataframe(lVanishMask)
                print("\n")
                print(f"\t-> Vanishing {lVanishMask[lVanishMask == True].size(0)} subnets: {(lVanishMask.nonzero().squeeze()+1).tolist()}")
                print(self._rules_dataframe)
            
            return True