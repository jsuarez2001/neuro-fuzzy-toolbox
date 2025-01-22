import torch

from neuro_fuzzy_toolbox.training import (
    Hybrid_learning_algorithm
)

class SONFIS(Hybrid_learning_algorithm):
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation=0, early_stopping=None, last_training_iteration=False):
        
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
        self.early_stopping = early_stopping
        
        # history
        self.history = {"loss": []}
        self.val_history = {"loss": []}
        
        
        # --------- ANFIS trainer ---------
        # Training parameters
        self.trainer_epochs = ANFIStrainer.epochs
        self.loss_function = ANFIStrainer.loss_function
        
        # Optimizer
        self.optimizer = ANFIStrainer.optimizer
        self.optimizer_params = ANFIStrainer.optimizer_params
        
        # Early stopping
        self.trainer_early_stopping = ANFIStrainer.early_stopping
        
        
        # ------ Internal variables ------
        self._freezed = torch.tensor([], dtype=torch.int)
        self._ages = torch.tensor([], dtype=torch.int)
        self._premises_consequents_relation = torch.tensor([], dtype=torch.bool)
        

    def __call__(self, ANFISmodel, loader, verbose=True):
        train_loader, val_loader = self._train_val_split(loader)
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        self._ages = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int)
        self._freezed = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int).bool()
        self._construct_premises_consequents_relation(ANFISmodel)
        
        self._parameters_update(ANFISmodel, train_loader, val_loader)
        if verbose:
            iter_width = len(str(self.max_iterations))
            if self.validation > 0:
                print(f'Iteration: {0:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
            else:
                print(f'Iteration: {0:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f}')
        
        model_updated = True
        i = 0
        while(model_updated and i < self.max_iterations):
            
            self._freeze_rules()
            
            # ANFIS structure updates (SONFIS operations)
            model_updated = self._structure_updates(ANFISmodel, train_loader)
            
            if verbose:
                print(f' -> Fuzzy rules: {ANFISmodel.fuzzy_rules}\n')
            
            if model_updated:
                self._parameters_update(ANFISmodel, train_loader, val_loader)
            else:
                if verbose:
                    print('No more updates')
                    
            self._register_loss(ANFISmodel, train_loader, val_loader)
            
            if self.validation > 0 and self._check_early_stop(ANFISmodel, self.val_history["loss"][-1], verbose):
                break
            
            if verbose:
                iter_width = len(str(self.max_iterations))
                if self.validation > 0:
                    print(f'Iteration: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
                else:
                    print(f'Iteration: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f}')
            
            i += 1
            
        self._unfreeze_rules()
        
        if self.last_training_iteration:
            self._parameters_update(ANFISmodel, train_loader, val_loader)
            
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        print('\nTraining finished')
        print(f' -> Fuzzy rules: {ANFISmodel.fuzzy_rules}\n')
            
    
    def _freeze_rules(self):
        self._freezed = torch.ones_like(self._freezed).bool()
        
    def _unfreeze_rules(self):
        self._freezed = torch.zeros_like(self._freezed).bool()

    
    def _parameters_update(self, ANFISmodel, train_loader, val_loader):
        ep = 0
        while ep < self.trainer_epochs:
            # Consequents update
            current_consequents = ANFISmodel.get_consequents()
            new_consequents = self._consequents_update(ANFISmodel, train_loader)
            
            freezed_consequents = self._get_consequent_rules_by_fuzzy_rules(self._freezed).sum(dim=0).bool()
            
            new_consequents[:, freezed_consequents, :] = current_consequents[:, freezed_consequents, :]
            ANFISmodel.set_consequents(new_consequents)
        
        
            # Premises update
            current_premises = ANFISmodel.get_premises()
            self._premises_update(ANFISmodel, train_loader)
            new_premises = ANFISmodel.get_premises()
        
            freezed_premises = self._freezed
            
            new_premises[:, freezed_premises, :] = current_premises[:, freezed_premises, :]
            ANFISmodel.set_premises(new_premises)
            
            if self.validation > 0 and self.trainer_early_stopping is not None:
                _, val_loss = self._loss(ANFISmodel, train_loader, val_loader)
                self.trainer_early_stopping(ANFISmodel, val_loss, False)
                if self.trainer_early_stopping._stop:
                    break
                
            ep += 1
            
        self.trainer_early_stopping.reset()
    
    
    
    def _check_early_stop(self, ANFISmodel, loss, verbose):
        if self.early_stopping is not None:
            self.early_stopping(ANFISmodel, loss, verbose)
            if self.early_stopping._stop:
                self._ages = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int)
                self._freezed = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int).bool()
                self._construct_premises_consequents_relation(ANFISmodel)
                return True
        return False
    
    def _construct_premises_consequents_relation(self, ANFISmodel):
        relations = torch.tensor([], dtype=torch.bool)
        fuzzy_rule = torch.zeros(ANFISmodel.fuzzy_rules).bool()
        for rule in range(ANFISmodel.fuzzy_rules):
            fuzzy_rule[rule] = True
            
            m_vals = ANFISmodel._fuzzification_layer(torch.ones(ANFISmodel._input_size).unsqueeze(0))
            m_vals = torch.ones_like(m_vals)
            m_vals[:, :, fuzzy_rule] = 0
            relations = torch.cat((relations, ~ANFISmodel._firing_levels_layer(m_vals).bool()), dim=0)
            
            fuzzy_rule[rule] = False
        self._premises_consequents_relation = relations.t()
        
    def _get_consequent_rules_by_fuzzy_rules(self, fuzzy_rules):
        # fuzzy_rules is a integer tensor
        # also works with boolean tensors, but in a diff way
        return self._premises_consequents_relation.t()[fuzzy_rules]
    
    def _get_fuzzy_rules_by_consequent_rules(self, consequent_rules):
        # consequent_rules is a integer tensor
        # also works with boolean tensors, but in a diff way
        return self._premises_consequents_relation[consequent_rules]
    
    
    def _structure_updates(self, ANFISmodel, train_loader):        
        # Grow
        did_Grow = self._GrowNet(ANFISmodel, train_loader)
        
        # Split
        if not did_Grow:
            did_Split = self._SplitSubNet(ANFISmodel, train_loader)
                    
        # Vanish
        did_Vanish = self._VanishNet(ANFISmodel, train_loader)
        
        return did_Grow or did_Split or did_Vanish
    
    
    def _GrowNet(self, ANFISmodel, train_loader):
        x = torch.tensor([])
        w_max = torch.tensor([])
        related_fuzzy_rules = torch.tensor([], dtype=torch.bool)
        for batch_x, _ in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            w_max = torch.cat((w_max, max_fl.values))
            x = torch.cat((x, batch_x))
            related_fuzzy_rules = torch.cat((related_fuzzy_rules, self._get_fuzzy_rules_by_consequent_rules(max_fl.indices)))
        
        dGrowMask = w_max <= self.dGrow**ANFISmodel._input_size
        
        bad_x = x[dGrowMask]
        bad_related_fuzzy_rules = related_fuzzy_rules[dGrowMask]
        
        Ngrow_mask = bad_related_fuzzy_rules.sum(dim=0) > self.Ngrow
        samples_used_mask = bad_related_fuzzy_rules.t()[Ngrow_mask].t()

        counts = samples_used_mask.sum(dim=0, keepdim=True).t()
        reshaped_bad_x = bad_x.unsqueeze(1)
        
        means = (reshaped_bad_x * samples_used_mask.unsqueeze(-1)).sum(dim=0) / counts
        stds = (((((reshaped_bad_x - means.unsqueeze(0))**2) * samples_used_mask.unsqueeze(-1)).sum(dim=0)) / counts).sqrt()
        
        n_new_fuzzy_rules = means.size(0)
        n_old_fuzzy_rules = ANFISmodel.fuzzy_rules
        
        if n_new_fuzzy_rules == 0:
            return False
        else:
            # Add new premises
            new_premises = ANFISmodel._fuzzification_layer._membership_function._grow_new_premise_parameters(means, stds)
            new_premises = torch.cat((ANFISmodel.get_premises(), new_premises), dim=1)
            ANFISmodel.set_premises(new_premises)
            
            # Update premises-consequents relation
            self._construct_premises_consequents_relation(ANFISmodel)
            
            # Add new consequents
            n_new_consequents = 0
            if ANFISmodel._rule_reduced:
                n_new_consequents = ANFISmodel.fuzzy_rules
            else:
                n_new_consequents = ANFISmodel.fuzzy_rules**ANFISmodel._input_size
                
            new_consequents = ANFISmodel._consequent_layer._consequent_function.initialize_consequents(ANFISmodel._outputs, n_new_consequents, ANFISmodel._input_size, ANFISmodel._dtype)
            old_consequents_positions = ~self._get_consequent_rules_by_fuzzy_rules(torch.cat((torch.zeros(n_old_fuzzy_rules), torch.ones(n_new_fuzzy_rules))).bool()).sum(dim=0).bool()
            new_consequents[:, old_consequents_positions, :] = ANFISmodel.get_consequents()
            ANFISmodel.set_consequents(new_consequents)
            
            # Update ages and freezed
            self._ages = torch.cat((self._ages, torch.zeros(n_new_fuzzy_rules, dtype=torch.int)))
            self._freezed = torch.cat((self._freezed, torch.zeros(n_new_fuzzy_rules, dtype=torch.int).bool()))
            
            return True
    
    
    def _SplitSubNet(self, ANFISmodel, train_loader):
        y = torch.tensor([])
        related_fuzzy_rules = torch.tensor([], dtype=torch.bool)
        model_outputs = torch.tensor([])
        for batch_x, batch_y in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            y = torch.cat((y, batch_y))
            model_outputs = torch.cat((model_outputs, ANFISmodel(batch_x, return_probabilities=True)))
            related_fuzzy_rules = torch.cat((related_fuzzy_rules, self._get_fuzzy_rules_by_consequent_rules(max_fl.indices)))
        
        """
        premilinary fix for multiclass output
        """
        if y.shape != model_outputs.shape:
            y = torch.nn.functional.one_hot(y.to(torch.long), ANFISmodel._outputs)
        """
        premilinary fix for multiclass output
        """
        
        expanded_related_fuzzy_rules = related_fuzzy_rules.t().unsqueeze(-1)
        
        masked_y = expanded_related_fuzzy_rules * y
        masked_outputs = expanded_related_fuzzy_rules * model_outputs
        counts = related_fuzzy_rules.sum(dim=0)
        mse_counts = counts*ANFISmodel._input_size
        
        mse_by_group = (((masked_y - masked_outputs) ** 2).sum(dim=1).sum(dim=1))/mse_counts
        
        NsplitMask = counts > self.Nsplit
        eSplitMask = mse_by_group > self.eSplit
        
        fuzzy_rules_to_split = NsplitMask*eSplitMask
        
        
        if fuzzy_rules_to_split[fuzzy_rules_to_split == True].size(0) == 0:
            False
        
        else:
            self._ages = self._ages[~fuzzy_rules_to_split]
            self._freezed = self._freezed[~fuzzy_rules_to_split]
            
            # Split premises
            still_premises = ANFISmodel.get_premises()[:, ~fuzzy_rules_to_split, :]
            split_premises = ANFISmodel._fuzzification_layer._membership_function._split_premise_parameters(ANFISmodel.get_premises()[:, fuzzy_rules_to_split, :])
            new_premises = torch.cat((still_premises, split_premises), dim=1)
            
            still_consequents = ~self._get_consequent_rules_by_fuzzy_rules(fuzzy_rules_to_split).sum(dim=0).bool()
            still_consequents = ANFISmodel.get_consequents()[:, still_consequents, :]

            ANFISmodel.set_premises(new_premises)
            
            # Update premises-consequents relation
            self._construct_premises_consequents_relation(ANFISmodel)
            
            # Add new consequents
            n_new_consequents = 0
            if ANFISmodel._rule_reduced:
                n_new_consequents = ANFISmodel.fuzzy_rules
            else:
                n_new_consequents = ANFISmodel.fuzzy_rules**ANFISmodel._input_size
                
            n_still_fuzzy_rules = still_premises.size(1)
            n_new_fuzzy_rules = split_premises.size(1)
            new_fuzzy_rules = torch.cat((torch.zeros(n_still_fuzzy_rules), torch.ones(n_new_fuzzy_rules))).bool()
            
            new_consequents = ANFISmodel._consequent_layer._consequent_function.initialize_consequents(ANFISmodel._outputs, n_new_consequents, ANFISmodel._input_size, ANFISmodel._dtype)    
            old_consequents_positions = ~self._get_consequent_rules_by_fuzzy_rules(new_fuzzy_rules).sum(dim=0).bool()
            
            new_consequents[:, old_consequents_positions, :] = still_consequents
            ANFISmodel.set_consequents(new_consequents)
            
            # Update ages and freezed
            new_ages = torch.zeros(fuzzy_rules_to_split[fuzzy_rules_to_split == True].size(0) * 2, dtype=torch.int)
            self._ages = torch.cat((self._ages, new_ages))
            
            new_freezed = torch.zeros(fuzzy_rules_to_split[fuzzy_rules_to_split == True].size(0) * 2, dtype=torch.int).bool()
            self._freezed = torch.cat((self._freezed, new_freezed))
            
            return True
    
    
    def _VanishNet(self, ANFISmodel, train_loader):
        related_fuzzy_rules = torch.tensor([], dtype=torch.bool)
        for batch_x, _ in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            related_fuzzy_rules = torch.cat((related_fuzzy_rules, self._get_fuzzy_rules_by_consequent_rules(max_fl.indices)))
        
        counts = related_fuzzy_rules.sum(dim=0)
        
        ages_update = (counts < self.Nvanish).int()
        self._ages *= ages_update
        self._ages += ages_update
        
        lVanishMask = self._ages > self.lVanish
        
        if lVanishMask.sum() == 0:
            return False
        
        else:
            new_premises = ANFISmodel.get_premises()[:, ~lVanishMask, :]
            
            consequents_to_vanish = self._get_consequent_rules_by_fuzzy_rules(lVanishMask).sum(dim=0).bool()
            new_consequents = ANFISmodel.get_consequents()[:, ~consequents_to_vanish, :]
            
            self._ages = self._ages[~lVanishMask]
            self._freezed = self._freezed[~lVanishMask]
            
            ANFISmodel.set_premises(new_premises)
            
            # Update premises-consequents relation
            self._construct_premises_consequents_relation(ANFISmodel)
            
            ANFISmodel.set_consequents(new_consequents)
            
            return True
        

class rule_reduced_SONFIS(SONFIS):
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation=0, early_stopping=None, last_training_iteration=False):
        super().__init__(Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation, early_stopping, last_training_iteration)
        self._last_best_rules = torch.tensor([-1])
    
    
    def __call__(self, ANFISmodel, loader, verbose=True):
        if ANFISmodel._rule_reduced == False:
            raise ValueError('The ANFIS model must be rule reduced')
            
        train_loader, val_loader = self._train_val_split(loader)
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        self._ages = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int)
        self._freezed = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int).bool()
        
        self._parameters_update(ANFISmodel, train_loader, val_loader)
        if verbose:
            iter_width = len(str(self.max_iterations))
            if self.validation > 0:
                print(f'Iteration: {0:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
            else:
                print(f'Iteration: {0:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f}')
        
        
        model_updated = True
        i = 0
        while(model_updated and i < self.max_iterations):
            
            self._freeze_rules()
            
            model_updated = self._structure_updates(ANFISmodel, train_loader)
            
            if verbose:
                print(f' -> Fuzzy rules: {ANFISmodel.fuzzy_rules}\n')
            
            if model_updated:
                self._parameters_update(ANFISmodel, train_loader, val_loader)
            else:
                if verbose:
                    print('No more updates')
                    
            self._register_loss(ANFISmodel, train_loader, val_loader)
            
            if self.validation > 0 and self._check_early_stop(ANFISmodel, self.val_history["loss"][-1], verbose):
                break
            
            if verbose:
                iter_width = len(str(self.max_iterations))
                if self.validation > 0:
                    print(f'Iteration: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
                else:
                    print(f'Iteration: {i+1:{iter_width}}/{self.max_iterations} - loss: {self.history["loss"][-1]:.6f}')
            
            i += 1
            
        self._unfreeze_rules()
        
        if self.last_training_iteration:
            self._parameters_update(ANFISmodel, train_loader, val_loader)
            
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        print('\nTraining finished')
        print(f' -> Fuzzy rules: {ANFISmodel.fuzzy_rules}\n')
        
    def _parameters_update(self, ANFISmodel, train_loader, val_loader):
        ep = 0
        while ep < self.trainer_epochs:
            # Consequents update
            current_consequents = ANFISmodel.get_consequents()
            new_consequents = self._consequents_update(ANFISmodel, train_loader)
            
            freezed_consequents = self._freezed
            
            new_consequents[:, freezed_consequents, :] = current_consequents[:, freezed_consequents, :]
            ANFISmodel.set_consequents(new_consequents)
        
        
            # Premises update
            current_premises = ANFISmodel.get_premises()
            self._premises_update(ANFISmodel, train_loader)
            new_premises = ANFISmodel.get_premises()
        
            freezed_premises = self._freezed
            
            new_premises[:, freezed_premises, :] = current_premises[:, freezed_premises, :]
            ANFISmodel.set_premises(new_premises)
            
            if self.validation > 0 and self.trainer_early_stopping is not None:
                _, val_loss = self._loss(ANFISmodel, train_loader, val_loader)
                self.trainer_early_stopping(ANFISmodel, val_loss, False)
                if self.trainer_early_stopping._stop:
                    break
                
            ep += 1
            
        self.trainer_early_stopping.reset()        
    
    def _structure_updates(self, ANFISmodel, train_loader):
        did_Grow = self._GrowNet(ANFISmodel, train_loader)
        
        if not did_Grow:
            did_Split = self._SplitSubNet(ANFISmodel, train_loader)
        
        did_Vanish = self._VanishNet(ANFISmodel, train_loader)
        
        return did_Grow or did_Split or did_Vanish
    
    def _check_early_stop(self, ANFISmodel, loss, verbose):
        if self.early_stopping is not None:
            self.early_stopping(ANFISmodel, loss, verbose)
            if self.early_stopping._stop:
                self._ages = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int)
                self._freezed = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int).bool()
                return True
        return False
    
    def _GrowNet(self, ANFISmodel, train_loader):
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
            fuzzy_rules = [best_bs_rules == rule for rule in torch.unique(best_bs_rules)]
            
            means = torch.stack([bad_samples[fuzzy_rule].mean(dim=0) for fuzzy_rule in fuzzy_rules])
            stds = torch.stack([bad_samples[fuzzy_rule].std(dim=0) for fuzzy_rule in fuzzy_rules])
            
            new_premises = ANFISmodel._fuzzification_layer._membership_function._grow_new_premise_parameters(means, stds)
            ANFISmodel.set_premises(torch.cat((ANFISmodel.get_premises(), new_premises), dim=1))
            
            n_new_consequents = new_premises.shape[1]
            new_consequents = ANFISmodel._consequent_layer._consequent_function.initialize_consequents(ANFISmodel._outputs, n_new_consequents, ANFISmodel._input_size, ANFISmodel._dtype)
            ANFISmodel.set_consequents(torch.cat((ANFISmodel.get_consequents(), new_consequents), dim=1))
            
            self._ages = torch.cat((self._ages, torch.zeros(new_premises.shape[1], dtype=torch.int)))
            self._freezed = torch.cat((self._freezed, torch.zeros(new_premises.shape[1], dtype=torch.int).bool()))
            
            return True
            
        
    def _SplitSubNet(self, ANFISmodel, train_loader):
        targets = torch.tensor([])
        model_outputs = torch.tensor([])
        best_rules = torch.tensor([], dtype=torch.int)

        for batch_x, batch_y in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            
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
            
            mse_values = torch.stack([((targets[best_rules == rule] - model_outputs[best_rules == rule])**2).mean() for rule in unique_rules])
            
            eSplit_mask = mse_values > self.eSplit
            
            if ((targets.shape[0] == 0) or (unique_rules[eSplit_mask].shape[0] == 0)):
                return False
            
            else:
                new_premises = ANFISmodel.get_premises()
                new_consequents = ANFISmodel.get_consequents()
                
                for rule in list(torch.flip(unique_rules[eSplit_mask], [0]).long()):
                    new_premises = torch.cat((new_premises[:, :rule,:], new_premises[:, rule+1:, :]), dim=1)
                    to_split = ANFISmodel.get_premises()[:, rule:rule+1, :]
                    split = ANFISmodel._fuzzification_layer._membership_function._split_premise_parameters(to_split)
                    
                    new_premises = torch.cat((new_premises, split), dim=1)
                    
                    new_consequents = torch.cat((new_consequents[:, :rule, :], new_consequents[:, rule+1:, :]), dim=1)
                    new_consequent = ANFISmodel._consequent_layer._consequent_function.initialize_consequents(ANFISmodel._outputs, 2, ANFISmodel._input_size, ANFISmodel._dtype)
                    new_consequents = torch.cat((new_consequents, new_consequent), dim=1)
                    
                    self._ages = torch.cat((self._ages[:rule], self._ages[rule+1:]))
                    self._freezed = torch.cat((self._freezed[:rule], self._freezed[rule+1:]))
                    self._ages = torch.cat((self._ages, torch.zeros(2, dtype=torch.int)))
                    self._freezed = torch.cat((self._freezed, torch.zeros(2, dtype=torch.int).bool()))
                    
                ANFISmodel.set_premises(new_premises)
                ANFISmodel.set_consequents(new_consequents)
                
                return True
    
    def _VanishNet(self, ANFISmodel, train_loader):
        best_rules = torch.tensor([], dtype=torch.int)
        
        for batch_x, _ in train_loader:
            firing_levels, _, _ = ANFISmodel.intermediate_values(batch_x)
            max_fl = torch.max(firing_levels, dim=1)
            
            best_rules = torch.cat((best_rules, max_fl.indices), dim=0)
            
        unique_rules, counts = torch.unique(best_rules, return_counts=True)
        all_rules = torch.arange(ANFISmodel.fuzzy_rules)
        
        total_counts = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int64)
        total_counts[unique_rules] = counts
        
        if torch.equal(best_rules, self._last_best_rules) or torch.equal(self._last_best_rules, torch.tensor([-1])):
            self._ages += 1
        else:
            last_unique_rules, last_counts = torch.unique(self._last_best_rules, return_counts=True)
            last_total_counts = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int64)
            last_total_counts[last_unique_rules[last_unique_rules < ANFISmodel.fuzzy_rules]] = last_counts[last_unique_rules < ANFISmodel.fuzzy_rules]
            
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
            
            self._last_best_rules = best_rules
            return True
        

class alt_SONFIS(SONFIS):
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation=0, early_stopping=None, last_training_iteration=False):
        super().__init__(Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, validation, early_stopping, last_training_iteration)
        self._last_best_rules = torch.tensor([-1])
        
    def _GrowNet(self, ANFISmodel, train_loader):
        pass
