import torch

from neuro_fuzzy_toolbox.training import (
    base_model_trainer
)
from neuro_fuzzy_toolbox.models.anfis import rule_reduced_ANFIS
from neuro_fuzzy_toolbox.utils._loader import get_loader_tensors
from neuro_fuzzy_toolbox.utils._least_squares import _rules_consequents_estimation, _local_consequents_estimation
from neuro_fuzzy_toolbox.utils._targets import _compute_loss

import pandas as pd
import numpy as np

class SONFIS(base_model_trainer):
    """
    Self-Organizing Neuro-Fuzzy Inference System (SONFIS) algorithm.
    
    Combines a parameter learning algorithm with three structural adaptation
    operators (GrowNet, SplitNet, and VanishNet) to iteratively update
    both the parameters and the structure of a rule-reduced ANFIS model.
    
    Note:
        This algorithm is only applicable to instances of :class:`rule_reduced_ANFIS`.
    """
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIStrainer, early_stopping=None, lse_for_new_consequents=None, lse_for_new_consequents_samples='origin', lse_for_new_consequents_lambda=0., last_training_iteration=False):
        """
        Initializes a new SONFIS instance.
        
        Args:
            Ngrow (int): Minimum number of poorly modeled samples required to grow a new subnet.
            dGrow (float): Threshold for identifying poorly modeled samples. A sample is considered poorly modeled if
                its maximum firing level across all subnets is less than or equal to this value raised to the power of
                the input dimensionality.
            Nsplit (int): Minimum number of samples associated with a subnet for it to be considered for splitting.
            eSplit (float): Minimum loss value of the samples associated with a subnet for it to be considered for splitting.
            Nvanish (int): Number of associated samples below which the age counter of a subnet is incremented (the counter is
                reset to ``0`` otherwise). With ``Nvanish=1``, the age only increases when the subnet has no associated samples.
            lVanish (int): Age at which a subnet is removed: a subnet is removed as soon as its age reaches ``lVanish``
                (threshold :math:`\\lambda` of the original formulation).
            max_iterations (int): Maximum number of structural adaptation iterations.
            ANFIStrainer (base_model_trainer): Instantiated ANFIS training algorithm that defines how the model parameters
                are updated at each iteration.
            early_stopping (EarlyStopping): Early stopping mechanism to use during the SONFIS iterations, based on the validation
                loss. The initial model (after its first training) is also a candidate to be restored. Defaults to ``None``.
            lse_for_new_consequents (str | None): Least-squares initialization of the consequent parameters of the rules
                created by GrowNet and SplitNet, which are otherwise initialized randomly. With ``None``, no estimation is
                performed (random initialization, as in the original formulation). With ``'isolated'``, the new rules are
                estimated jointly against the targets, ignoring the contribution of the existing rules. With ``'local'``,
                each new rule is estimated separately by weighted least squares, weighted by its (unnormalized) firing
                level. With ``'global'``, the new rules are estimated jointly on the residual left by the existing rules.
                This option only matters with gradient-based trainers, since the hybrid trainer re-estimates the
                consequents in its first epoch. Defaults to ``None``.
            lse_for_new_consequents_samples (str): Samples used by that estimation. With ``'origin'``, each group of
                rules created from the same samples (one rule in GrowNet, the two halves of a split rule in SplitNet) is
                estimated only on those samples. With ``'all'``, all the new rules of the operator are estimated together
                on the whole training set. Defaults to ``'origin'``.
            lse_for_new_consequents_lambda (float): Lambda value for Ridge regularization in the least-squares initialization 
                of new consequent parameters. If ``0.``, no regularization is applied. Defaults to ``0.``.
            last_training_iteration (bool): If ``True``, performs a final parameter update over all subnets after the SONFIS 
                algorithm finishes. Defaults to ``False``.
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
        if lse_for_new_consequents not in (None, 'isolated', 'local', 'global'):
            raise ValueError(f"lse_for_new_consequents={lse_for_new_consequents!r} is not a valid option. "
                             "Use None, 'isolated', 'local' or 'global'.")
        if lse_for_new_consequents_samples not in ('origin', 'all'):
            raise ValueError(f"lse_for_new_consequents_samples={lse_for_new_consequents_samples!r} is not a valid option. "
                             "Use 'origin' or 'all'.")
        self.lse_for_new_consequents = lse_for_new_consequents
        self.lse_for_new_consequents_samples = lse_for_new_consequents_samples
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
        
        self._get_max_firing_level = None
        
        
        # ------ Rules Dataframe ------
        self._rules_dataframe = None
        self._current_best_rules_dataframe = None
        self._current_max_idx = 0
        self._best_rules_dataframe_iter = None
    
    
    def __call__(self, ANFISmodel, train_loader, val_loader=None, verbose=True):
        """
        Runs the SONFIS algorithm.

        At each iteration, the structural adaptation operators GrowNet,
        SplitNet, and VanishNet are applied, followed by a parameter update
        of the modified subnets. The algorithm stops when no structural updates
        occur or the maximum number of iterations is reached. If early stopping
        is enabled and a validation DataLoader is provided, the algorithm may
        also stop early based on the validation loss.

        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model to train.
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data. Defaults to ``None``.
            verbose (bool): If ``True``, prints progress and structural update messages at each iteration. Defaults to ``True``.
        """

        if not isinstance(ANFISmodel, rule_reduced_ANFIS):
            raise ValueError(
                f'SONFIS requires a rule_reduced_ANFIS model, but received a {type(ANFISmodel).__name__}. '
                'SONFIS updates only the parameters of the subnets that are not frozen, which requires premise and '
                'consequent parameters stored separately for each rule (this is not the case for ANFIS or h_ANFIS, '
                'even with rule_reduced=True).'
            )
        self._set_max_firing_level_method(ANFISmodel)
        
        self._current_max_idx = ANFISmodel.rules - 1
        
        self._ages = torch.zeros(ANFISmodel.rules, dtype=torch.int)
        self._freezed = torch.zeros(ANFISmodel.rules, dtype=torch.int).bool()
        
        use_early_stopping = (val_loader is not None) and (self.sonfis_early_stopping is not None)
        if use_early_stopping:
            self.sonfis_early_stopping.reset()
        early_stop_flag = False
        self._best_rules_dataframe_iter = 0
        
        iter_width = len(str(self.max_iterations))
        if verbose:
            print(f'ITERATION: {0:{iter_width}}/{self.max_iterations}')
        
        self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
        
        # The losses are registered after the initial training, so that the i-th element of the history
        # corresponds to the model obtained at the i-th iteration (the 0-th being the initial model).
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        if verbose:
            self._rules_dataframe = ANFISmodel.get_rules_structure().reset_index(drop=True)
            if val_loader is not None:
                self._current_best_rules_dataframe = self._rules_dataframe.copy()
            print("\nSTARTING STATE:")
            print(self._rules_dataframe.to_string())
            self._print_loss(ANFISmodel, val_loader)
        
        # The initial model is the first candidate of the early stopping mechanism
        if use_early_stopping:
            self.sonfis_early_stopping(ANFISmodel, self.val_history["loss"][-1])
        
        model_updated = True
        i = 0
        while(model_updated and i < self.max_iterations):
            
            if verbose:
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
                
                if verbose:
                    self._print_loss(ANFISmodel, val_loader)
            
                if use_early_stopping:
                    if self._check_early_stop(ANFISmodel, self.val_history["loss"][-1], verbose):
                        if verbose:
                            print(f'found on the {self._best_rules_dataframe_iter}° iteration.')
                        early_stop_flag = True
                        if verbose:
                            # The restored model is the best one: its rules keep their identifiers
                            self._rules_dataframe = self._current_best_rules_dataframe.copy()
                        break
                    elif self.sonfis_early_stopping._counter == 0:
                        self._best_rules_dataframe_iter = i+1
                        if verbose:
                            self._current_best_rules_dataframe = self._rules_dataframe.copy()
            
            elif verbose:
                print('NO MORE UPDATES')
            
            i += 1
            
        if verbose and i == self.max_iterations:
            print('MAX ITERATIONS REACHED')
            
        self._unfreeze_subnets()
        
        if self.last_training_iteration:
            if verbose:
                print('\nLast training iteration (all subnets are being trained again)')
            
            self.trainer._sonfis_update_parameters(ANFISmodel, train_loader, val_loader, self._freezed)
            
            if verbose:
                self._replace_trained_subnets_on_rules_dataframe(ANFISmodel)
                print("LAST TRAINING STATE:")
                print(self._rules_dataframe.to_string())
            
            self._register_loss(ANFISmodel, train_loader, val_loader)

            if verbose:
                self._print_loss(ANFISmodel, val_loader, show_rules=False, leading_newline=False)
            
        if verbose:
            print('\nTRAINING FINISHED')
            print(f'\t --> ANFIS rules: {ANFISmodel.rules}\n')
            print(self._rules_dataframe.to_string())
        
        
    # ----- Freezed subnets -----
    def _freeze_subnets(self):
        """
        Freezes all subnets, preventing their parameters from being updated during the next training step.
        """
        self._freezed = torch.ones_like(self._freezed).bool()
        
    def _unfreeze_subnets(self):
        """
        Unfreezes all subnets, allowing their parameters to be updated during the next training step.
        """
        self._freezed = torch.zeros_like(self._freezed).bool()
    
    
    # ----- Early Stopping -----
    def _print_loss(self, ANFISmodel, val_loader, show_rules=True, leading_newline=True):
        """
        Prints the last registered training (and validation) loss and, optionally, the current number of rules.

        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model being trained.
            val_loader (DataLoader | None): DataLoader containing the validation data.
            show_rules (bool): If ``True``, also prints the number of rules. Defaults to ``True``.
            leading_newline (bool): If ``True``, prints an empty line before the loss. Defaults to ``True``.
        """
        prefix = '\n\t' if leading_newline else '\t'
        if val_loader is not None:
            print(f'{prefix}loss: {self.history["loss"][-1]:.6f} - validation loss: {self.val_history["loss"][-1]:.6f}')
        else:
            print(f'{prefix}loss: {self.history["loss"][-1]:.6f}')
        if show_rules:
            print(f'\t --> ANFIS rules: {ANFISmodel.rules}\n')

    def _check_early_stop(self, ANFISmodel, loss, verbose=True):
        """
        Checks whether the SONFIS early stopping criterion is met.

        If early stopping is triggered, the subnet ages and frozen states are
        reset and the early stopping mechanism is reinitialized.

        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model being trained.
            loss (float): Current validation loss value.
            verbose (bool): If ``True``, prints a message when early stopping is triggered. Defaults to ``True``.

        Returns:
            bool: ``True`` if training should be stopped, ``False`` otherwise.
        """
        if self.sonfis_early_stopping is not None:
            self.sonfis_early_stopping(ANFISmodel, loss, verbose=verbose)
            if self.sonfis_early_stopping.stop:
                self._ages = torch.zeros(ANFISmodel.rules, dtype=torch.int)
                self._freezed = torch.zeros(ANFISmodel.rules, dtype=torch.int).bool()
                self.sonfis_early_stopping.reset()
                return True
        return False
    
    
    # ----- Rules dataframe -----
    def _add_subnets_on_rules_dataframe(self, new_premises, new_consequents):
        """
        Appends rows for newly added subnets to the internal rules DataFrame.

        Args:
            new_premises (torch.Tensor): Premise parameters of the new subnets, of shape ``(input_size, n_new_subnets, mf_params)``.
            new_consequents (torch.Tensor): Consequent parameters of the new subnets, of shape ``(outputs, n_new_subnets, input_size + 1)``.
        """
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
        """
        Removes rows corresponding to subnets indicated by a boolean mask
        from the internal rules DataFrame.

        Args:
            mask (torch.Tensor): Boolean tensor of length ``num_rules``, where ``True`` indicates that the corresponding 
                subnet should be removed.
        """
        keep_mask = ~mask.cpu().numpy()
        
        self._rules_dataframe = self._rules_dataframe.loc[keep_mask]
        
    def _drop_subnets_on_rules_dataframe_by_idxs(self, idxs_tensor):
        """
        Removes rows corresponding to subnets at the specified indices from
        the internal rules DataFrame.

        Args:
            idxs_tensor (torch.Tensor): Tensor of integer indices indicating
                which subnets should be removed.
        """
        idxs = idxs_tensor.cpu().numpy()
        keep = np.ones(len(self._rules_dataframe), dtype=bool)
        keep[idxs] = False
        
        self._rules_dataframe = self._rules_dataframe.iloc[keep]
        
    def _replace_trained_subnets_on_rules_dataframe(self, ANFISmodel):
        """
        Updates the rows of the internal rules DataFrame corresponding to
        subnets that were not frozen during the last training step, reflecting
        their current parameter values.

        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model whose
                updated subnet parameters are used.
        """
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
        Assigns the method used to retrieve the maximum firing level per sample,
        depending on whether the model uses a default rule.
        
        If the model has a default rule, the last firing level (corresponding
        to the default rule) is excluded from the maximum computation.
        
        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model being trained.
        """
        if ANFISmodel._default_rule:
            self._get_max_firing_level = lambda firing_levels: torch.max(firing_levels[:, :-1], dim=1)
        else: 
            self._get_max_firing_level = lambda firing_levels: torch.max(firing_levels, dim=1)
        
    
    
    # ----- Update structure -----
    def _update_structure(self, ANFISmodel, train_loader, verbose):
        """
        Executes the GrowNet, SplitNet, and VanishNet structural adaptation
        operators in sequence.
        
        GrowNet is attempted first. If no new subnets are grown, SplitNet
        is attempted. VanishNet is always applied regardless of the outcomes
        of the previous operators.
        
        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model to update.
            train_loader (DataLoader): DataLoader containing the training data.
            verbose (bool): If ``True``, prints messages about structural changes.

        Returns:
            bool: ``True`` if any structural update was performed, ``False`` otherwise.
        """
        did_Grow = self._GrowNet(ANFISmodel, train_loader, verbose)
        
        did_Split = False
        if not did_Grow:
            did_Split = self._SplitNet(ANFISmodel, train_loader, verbose)
        
        did_Vanish = self._VanishNet(ANFISmodel, train_loader, verbose)

        return did_Grow or did_Split or did_Vanish
    
    
    def _GrowNet(self, ANFISmodel, train_loader, verbose):
        """
        Executes the GrowNet operator to add new subnets to the model.
        
        Identifies poorly modeled samples — those whose maximum firing level
        across all subnets is below ``dGrow ** input_size`` — and groups them
        by their associated subnet. For each group with more than ``Ngrow``
        samples, a new subnet is created, whose MFs are initialized with the
        mean and the (population) standard deviation of those samples.
        If ``lse_for_new_consequents`` is set, the consequent parameters
        of the new subnets are estimated by least squares
        (see :meth:`_init_new_consequents`).
        
        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model to update.
            train_loader (DataLoader): DataLoader containing the training data.
            verbose (bool): If ``True``, prints messages about the new subnets added.
        
        Returns:
            bool: ``True`` if at least one new subnet was added, ``False`` otherwise.
        """
        X, y = get_loader_tensors(train_loader)
        
        firing_levels = ANFISmodel.get_firing_levels(X)
        max_fl = self._get_max_firing_level(firing_levels)
        
        dGrowMask = max_fl.values <= self.dGrow**ANFISmodel._input_size # using dGrow
        
        bad_samples = X[dGrowMask]
        bad_targets = y[dGrowMask]
        clusters = max_fl.indices[dGrowMask] # clusters are the associated "bad samples" to each subnet based on the max firing level
        
        unique_rules, counts = torch.unique(clusters, return_counts=True) # how many "max firing levels" do each of the subnets get considering only the "bad_samples"?
        Ngrow_mask = counts > self.Ngrow
        
        indices_to_keep = torch.isin(clusters, unique_rules[Ngrow_mask]).nonzero().squeeze()  # using Ngrow
        
        bad_samples = bad_samples[indices_to_keep] # getting which samples will be considered
        bad_targets = bad_targets[indices_to_keep]
        clusters = clusters[indices_to_keep] # & the associated subnet
        
        if bad_samples.size(0) == 0:
            return False
        
        else:
            rules = [clusters == rule for rule in torch.unique(clusters)] # list of boolean masks (lenght: current number of subnets), each one with shape: (bad_samples.shape[0], ) 
            
            means = torch.stack([bad_samples[rule].mean(dim=0) for rule in rules]) # shape = (new_subnets, input_dim)
            stds = torch.stack([bad_samples[rule].std(dim=0, correction=0) for rule in rules]) # population std (as in the original formulation), shape = (new_subnets, input_dim)
            
            new_premises = ANFISmodel._fuzzification_layer._membership_function._grow_new_premise_parameters(means, stds)
            ANFISmodel.set_premises(torch.cat((ANFISmodel.get_premises(), new_premises), dim=1))
            
            n_new_rules = new_premises.shape[1]
            new_consequents = ANFISmodel._consequent_layer._consequent_function.random_consequents(ANFISmodel._outputs, n_new_rules, ANFISmodel._input_size, ANFISmodel._dtype)
            ANFISmodel.set_consequents(torch.cat((ANFISmodel.get_consequents(), new_consequents), dim=1))
            
            if self.lse_for_new_consequents is not None:
                first_new_rule = ANFISmodel.rules - n_new_rules
                groups = [(bad_samples[rule], bad_targets[rule], [first_new_rule + i]) for i, rule in enumerate(rules)]
                self._init_new_consequents(ANFISmodel, X, y, groups)
                new_consequents = ANFISmodel.get_consequents()[:, -n_new_rules:, :]
            
            if verbose:
                self._add_subnets_on_rules_dataframe(new_premises, new_consequents)
                print(f"\t-> Growing {n_new_rules} new subnets: {[i for i in range(self._current_max_idx - new_premises.shape[1] + 1, self._current_max_idx + 1)]}")
            
            self._ages = torch.cat((self._ages, torch.zeros(new_premises.shape[1], dtype=torch.int)))
            self._freezed = torch.cat((self._freezed, torch.zeros(new_premises.shape[1], dtype=torch.int).bool()))
            
            return True
        
    def _init_new_consequents(self, ANFISmodel, X, y, groups):
        """
        Initializes by least squares the consequent parameters of the subnets created by GrowNet or SplitNet,
        according to ``lse_for_new_consequents`` and ``lse_for_new_consequents_samples``.

        The existing subnets (all those not created by the operator) are never modified. With ``'global'``, their
        contribution is subtracted from the targets, while the other new subnets do not contribute.

        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model being trained, already containing the new subnets.
            X (torch.Tensor): Whole training inputs, of shape ``(n_samples, input_size)``.
            y (torch.Tensor): Whole training targets.
            groups (list[tuple]): One ``(samples, targets, rules)`` tuple per group of new subnets created from the same
                samples, where ``rules`` are the indices of those subnets in the model.
        """
        new_rules = sorted(r for _, _, rules in groups for r in rules)
        existing_rules = [r for r in range(ANFISmodel.rules) if r not in set(new_rules)]
        
        if self.lse_for_new_consequents_samples == 'origin':
            units = groups
        else:
            units = [(X, y, new_rules)]
        
        lam, driver = self.lse_for_new_consequents_lambda, "gelsd"
        consequents = ANFISmodel.get_consequents().clone()
        for x_unit, y_unit, rules in units:
            if self.lse_for_new_consequents == 'local':
                frozen = torch.ones(ANFISmodel.rules, dtype=torch.bool)
                frozen[rules] = False
                estimated = _local_consequents_estimation(ANFISmodel, x_unit, y_unit, driver, lam, frozen)[:, rules, :]
            else:
                residual_of = existing_rules if self.lse_for_new_consequents == 'global' else None
                estimated = _rules_consequents_estimation(ANFISmodel, x_unit, y_unit, rules, driver, lam, residual_of)
            consequents[:, rules, :] = estimated.to(consequents.dtype)
        
        ANFISmodel.set_consequents(consequents)
    
    def _SplitNet(self, ANFISmodel, train_loader, verbose):
        """
        Executes the SplitNet operator to split subnets with high local
        error.
        
        Identifies subnets that have more than ``Nsplit`` associated samples
        and a loss value above ``eSplit``, and splits each of them into two
        new subnets. If ``lse_for_new_consequents`` is set, the consequent
        parameters of the resulting subnets are estimated by least squares
        once all the splits are done (see :meth:`_init_new_consequents`).
        
        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model to update.
            train_loader (DataLoader): DataLoader containing the training data.
            verbose (bool): If ``True``, prints messages about the subnets split.
        
        Returns:
            bool: ``True`` if at least one subnet was split, ``False`` otherwise.
        """
        samples, targets = get_loader_tensors(train_loader)
        X_all, y_all = samples, targets
        
        firing_levels = ANFISmodel.get_firing_levels(samples)
        max_fl = self._get_max_firing_level(firing_levels)
        
        best_rules = max_fl.indices
         
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
            
            rules = [best_rules == rule for rule in unique_rules] # list of boolean masks (lenght: current number of subnets), each one with shape: (bad_samples.shape[0], ) 
            
            model_outputs_list = [model_outputs[rule] for rule in rules]
            targets_list = [targets[rule] for rule in rules]
                
            # loss of the samples associated with each of the considered subnets
            loss_values = torch.stack([_compute_loss(ANFISmodel, self.loss_function, model_outputs_list[i], targets_list[i]) for i in range(len(rules))])
            
            eSplit_mask = loss_values > self.eSplit
            
            rules_to_split = unique_rules[eSplit_mask]
            
            if ((targets.shape[0] == 0) or (rules_to_split.shape[0] == 0)): # using eSplit
                return False
            
            else:
                if self.lse_for_new_consequents is not None:
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
                    
                    new_premises = ANFISmodel.get_premises()
                    new_consequents = ANFISmodel.get_consequents()
                        
                    idx += 1
                
                n_new_rules = 2 * rules_to_split.shape[0]
                if self.lse_for_new_consequents is not None:
                    # The halves of the j-th split subnet occupy positions first_new_rule + 2j and first_new_rule + 2j + 1
                    first_new_rule = ANFISmodel.rules - n_new_rules
                    groups = [(to_split_samples_list[j], to_split_targets_list[j], [first_new_rule + 2 * j, first_new_rule + 2 * j + 1])
                              for j in range(rules_to_split.shape[0])]
                    self._init_new_consequents(ANFISmodel, X_all, y_all, groups)
                    all_new_consequents = ANFISmodel.get_consequents()[:, -n_new_rules:, :]
                
                if verbose:
                    # Stable identifiers of the split subnets (the index of the rules DataFrame may hold Python or NumPy integers)
                    subnets = [int(self._rules_dataframe.index[i]) for i in rules_to_split.long().tolist()]
                    print(f"\t-> Splitting {rules_to_split.shape[0]} subnets: {subnets}")
                    
                    self._drop_subnets_on_rules_dataframe_by_idxs(rules_to_split)
                    self._add_subnets_on_rules_dataframe(all_new_premises, all_new_consequents)
                
                return True
            
    def _group_samples_for_lse_in_order(self, eSplit_mask, targets_list, samples, rules):
        """
        Groups samples and targets in reverse order for least-squares
        estimation during the SplitNet operation.
        
        Processes subnets in reverse order to match the order in which they
        are split, ensuring consistency between the sample groups and the
        subnets being updated.
        
        Args:
            eSplit_mask (torch.Tensor): Boolean tensor indicating which subnets exceed the ``eSplit`` threshold and are to be split.
            targets_list (list[torch.Tensor]): List of target tensors, one per subnet considered for splitting.
            samples (torch.Tensor): Input samples of the subnets considered for splitting, of shape ``(n_samples, input_size)``.
            rules (list[torch.Tensor]): List of boolean masks indicating which samples are associated with each subnet.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple containing two lists (the grouped input samples and their corresponding targets) ordered to match the reverse splitting sequence.
        """
        to_split_samples_list = []
        to_split_targets_list = []
        
        i = eSplit_mask.shape[0] - 1
        for boolean in torch.flip(eSplit_mask, [0]):
            if boolean:
                to_split_targets_list.append(targets_list[i])
                to_split_samples_list.append(samples[rules[i]])
            i -= 1
            
        return to_split_samples_list, to_split_targets_list
            
    def _VanishNet(self, ANFISmodel, train_loader, verbose):
        """
        Executes the VanishNet operator to remove underused subnets.

        Tracks the age of each subnet by incrementing its counter whenever
        fewer than ``Nvanish`` samples are associated with it, and resetting
        it otherwise. Subnets whose age reaches or exceeds ``lVanish`` and
        that still have fewer than ``Nvanish`` associated samples are removed
        from the model.
        
        Args:
            ANFISmodel (rule_reduced_ANFIS): Rule-reduced ANFIS model to update.
            train_loader (DataLoader): DataLoader containing the training data.
            verbose (bool): If ``True``, prints messages about the subnets removed.
        
        Returns:
            bool: ``True`` if at least one subnet was removed, ``False`` otherwise.
        """
        X, _ = get_loader_tensors(train_loader)
        
        firing_levels = ANFISmodel.get_firing_levels(X)
        max_fl = self._get_max_firing_level(firing_levels)
        
        best_rules = max_fl.indices
        
        unique_rules, counts = torch.unique(best_rules, return_counts=True) # how many "max firing levels" do each of the subnets get?
        all_rules = torch.arange(ANFISmodel.rules)
        
        total_counts = torch.zeros(ANFISmodel.rules, dtype=torch.int64)
        total_counts[unique_rules] = counts
        
        self._ages[all_rules[(total_counts < self.Nvanish)]] += 1 # add 1 age to the subnets that have less than Nvanish "associated" samples
        self._ages[all_rules[(total_counts >= self.Nvanish)]] = 0 # reset age to 0 for the subnets that have more than Nvanish "associated" samples

        mask = ((self._ages >= self.lVanish) & (total_counts < self.Nvanish)) # using Nvanish & lVanish --> to filter by age 6 by number of "associated" samples
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
                # Stable identifiers of the removed subnets (the index of the rules DataFrame may hold Python or NumPy integers)
                subnets = [int(self._rules_dataframe.index[i]) for i in rules_to_eliminate.tolist()]
                print(f"\t-> Vanishing {rules_to_eliminate.size(0)} subnets: {subnets}")
                self._drop_subnets_on_rules_dataframe(mask)
            
            return True