import torch
import torch.utils.data as data
import torch.nn as nn

import anfis.train as train


class SONFIS():
    '''
    Self-Organizing Algorithm for Neuro-Fuzzy Inference Systems (SONFIS) for online learning and adaptation.

    **Attributes for initialization:**
        
    .. attribute:: OLSinstance
    
        The OLS instance to be applied while the SONFIS algorithm is running.
        
        :type: OLS
    
    .. attribute:: Ngrow
    
        Threshold for growing a new rule based on firing levels.
        
        :type: int
        
    .. attribute:: dGrow
    
        Threshold for deciding whether to grow a new rule based on firing level spread.
        
        :type: float
    
    .. attribute:: Nsplit
    
        Threshold for splitting a sub-network based on rule contribution.
        
        :type: int
        
    .. attribute:: eSplit
    
        Threshold for deciding whether to split a sub-network based on rule error.
        
        :type: float
        
    .. attribute:: Nvanish
    
        Threshold for deciding whether to vanish a rule based on the number of samples.
        
        :type: int
        
    .. attribute:: lVanish
    
        Threshold for deciding whether to vanish a rule based on its age.
        
        :type: int
        
    .. attribute:: max
    
        Maximum number of iterations for SONFIS learning.
        
        :type: int

        
    .. attribute:: validation
    
        Proportion of the training data to use for validation.
        
        :type: float
        :default: 0
        
    .. attribute:: early_stopping
    
        EarlyStopping instance for the SONFIS algorithm. For default works with the validation split, but if the validation portion is not entered, it will work with the entire training dataset.
        
        :type: EarlyStopping
        :default: None
        
    .. attribute:: last_ols
    
        Boolean flag for apply a last OLS iteration with all the rules unfreezed after SONFIS execution finishes.
        
        :type: boolean
        :default: True
        
    .. attribute:: reset_ols_early_stopping
    
        Boolean flag for apply a last OLS iteration with all the rules unfreezed after SONFIS execution finishes.
        
        :type: boolean
        :default: True

    **Other attributes:**
        
    .. attribute:: freezed
    
        Tensor indicating which rules are frozen (whose parameters should not be updated).

        :type: torch.tensor
        
    .. attribute:: ages
    
        Tensor that contains the ages of each of the rules of the ANFIS model.

        :type: torch.tensor
        
    .. attribute:: last_best_rules

        Tensor containing which rules modeled the most samples in the last iteration (used in the VanishNet operator).
        
        :type: torch.tensor
        
    .. attribute:: loss_function
    
        Loss function (obtained from the OLS instance).
        
        :type: torch.nn.functional
        :default: nn.functional.mse_loss
        
    .. attribute:: train_history

        Training loss and various measures history.
        
        :type: torch.tensor
        
    .. attribute:: val_history
    
        Validation loss and various measures history.
        
        :type: torch.tensor

    **Example Usage:**
    
    .. code::
    
        >>> #dataloader and ANFIS model
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        >>> input_data = loader.dataset.tensors[0]
        >>> anfis_model = Type3ANFIS(input_data, init_rules=1)
        >>> #OLS
        >>> optimizer = torch.optim.AdamW
        >>> optim_params = {'lr': 0.001, 'weight_decay': 0.001}
        >>> ols_early_stopping = EarlyStopping(patience=5)
        >>> ols = OLS(epochs=15, optimizer=optimizer, optim_params=optim_params, validation=0.2, early_stopping=ols_early_stopping)
        >>> #SONFIS
        >>> SONFISearly_stopping = EarlyStopping(patience=10, delta=0.01)
        >>> sonfis = SONFIS(ols, Ngrow=30, dGrow=0.8, Nsplit=20, eSplit=0.7, Nvanish=8, lVanish=4, max=50, validation=0.25, early_stopping=SONFISearly_stopping, last_ols=False)
        >>> sonfis(anfis_model, loader)
        >>> print(sonfis.history['loss'])
    
    **Methods:**

    '''
    def __init__(self, OLS_instance, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max, validation=0, early_stopping=None, last_ols=False, reset_ols_early_stopping=True):
        """
        Initializes a new SONFIS instance.

        Parameters:
        - max (int): Maximum number of iterations for SONFIS learning (default: 100).
        - y (float): Learning rate for OLS optimizer (default: 0.01).
        - OLSepochs (int): Number of epochs for OLS optimizer during each SONFIS iteration (default: 1).
        - Ngrow (int): Threshold for growing a new rule based on firing levels (default: 1).
        - dGrow (float): Threshold for deciding whether to grow a new rule based on firing level spread (default: 1.0).
        - Nsplit (int): Threshold for splitting a sub-network based on rule contribution (default: 1).
        - eSplit (float): Threshold for deciding whether to split a sub-network based on rule error (default: 0.1).
        - Nvanish (int): Threshold for deciding whether to vanish a rule based on the number of samples (default: 1).
        - lVanish (int): Threshold for deciding whether to vanish a rule based on age (default: 10).
        - loss_function (torch.nn.Module): Loss function for training (default: nn.functional.mse_loss).
        - validation (float): Proportion of the training data to use for validation (default: 0).

        """
        # Hyperparameters
        self.max = max
        self.Ngrow = Ngrow
        self.dGrow = dGrow
        self.Nsplit = Nsplit
        self.eSplit = eSplit
        self.Nvanish = Nvanish
        self.lVanish = lVanish

        #OLS
        self.ols = OLS_instance
        self.reset_ols_early_stopping = reset_ols_early_stopping

        #Other attributes
        self.last_ols = last_ols
        self.freezed = torch.tensor([], dtype=torch.int)
        self.ages = torch.tensor([], dtype=torch.int)
        self.last_best_rules = torch.tensor([-1], dtype=torch.int)

        #history
        self.loss_function = OLS_instance.loss_function

        self.history = {'loss': torch.tensor([])}
        self.val_history = {'loss': torch.tensor([])}
        
        #early stopping
        self.validation = validation
        self.early_stopping = early_stopping


    def __call__(self, ANFISmodel, loader):
        """
        Performs SONFIS learning on the provided ANFIS model and data loader.
        
        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        """
        train_loader, val_loader = self.val_partition(loader)
        _ = self.obtain_metrics(ANFISmodel, train_loader, val_loader)

        self.ages = torch.zeros(ANFISmodel.rules, dtype=torch.int)
        self.freezed = torch.zeros(ANFISmodel.rules, dtype=torch.int)

        model_updated = True
        i = 0

        self.callOLS(ANFISmodel, loader)
        while(model_updated & (i < self.max)):
            print("\n *******ITERATION:", i+1, " ******* ")
            self.freeze_rules()

            #GrowNet
            did_Grow = self.GrowNet(ANFISmodel, train_loader)

            #Split Sub-network
            if not did_Grow:
                did_Split = self.SplitSubNetwork(ANFISmodel, train_loader)
            else:
                did_Split = False

            #VanishNet
            did_Vanish = self.VanishNet(ANFISmodel, train_loader)

            print("\nRules amount:", ANFISmodel.rules, "\n\n")
            #Check update
            model_updated = did_Grow | did_Split | did_Vanish
            if model_updated:
                self.callOLS(ANFISmodel, loader)
            else:
                print("NO MORE UPDATES")

            loss = self.obtain_metrics(ANFISmodel, train_loader, val_loader)
            if self.callEarlyStopping(ANFISmodel, loss):
                break;

            i += 1

        self.unfreeze_rules()
        if self.last_ols:
            self.ols.last = True
            self.callOLS(ANFISmodel, loader)
        _ = self.obtain_metrics(ANFISmodel, train_loader, val_loader)
        
        
    def freeze_rules(self):
        """
        Freeze all rules in the ANFIS model.

        """
        self.freezed = torch.ones_like(self.freezed)
        
        
    def unfreeze_rules(self):
        """
        Unfreeze all rules in the ANFIS model.

        """
        self.freezed = torch.zeros_like(self.freezed)


    def callOLS(self, ANFISmodel, loader):
        """
        Calls the OLS optimizer to update the ANFIS model parameters.

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        """
        # Call OLS training routine
        self.ols(ANFISmodel, loader, self.freezed)

        # Reset OLS training history metrics
        for metric in self.ols.history:
            self.ols.history[metric] = torch.tensor([])

        # Reset OLS EarlyStopping if applicable
        if self.ols.early_stopping is not None and self.reset_ols_early_stopping:
            self.ols.early_stopping.reset()
            
    def callEarlyStopping(self, ANFISmodel, loss):
        """
        Calls the EarlyStopping mechanism and handles early stopping for the SONFIS model.

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        :return bool: True if early stopping criterion do establish and early stopping for the execution of the SONFIS algorithm, False otherwise (If there is none early stopping mechanism for the SONFIS execution, then just returns False).

    """
        if self.early_stopping != None:
            self.early_stopping(loss, ANFISmodel)
            if self.early_stopping.early_stop:
                print("EARLY STOPPING")
                self.freezed = self.freezed[:ANFISmodel.rules]
                self.ages = self.ages[:ANFISmodel.rules]
                return True
            return False
        return False
    
    
    def val_partition(self, loader):
        """
        Splits the provided DataLoader into training and validation sets based on the specified validation ratio.

        :param loader: DataLoader containing input data and labels.
        :type loader: torch.utils.data.DataLoader

        :return:
            - train_loader (torch.utils.data.DataLoader): DataLoader for training set.
            - val_loader (torch.utils.data.DataLoader): DataLoader for validation set.

        """
        if self.validation != 0:
            x_train, y_train = loader.dataset.tensors
            indices = torch.randperm(x_train.size(0))
            x_train = x_train[indices]
            y_train = y_train[indices]

            split_index = int(x_train.shape[0] * self.validation)

            x_val = x_train[:split_index]
            y_val = y_train[:split_index]
            x_train = x_train[split_index:]
            y_train = y_train[split_index:]

            train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size=loader.batch_size, shuffle=True)
            val_loader = data.DataLoader(data.TensorDataset(x_val, y_val), batch_size=loader.batch_size, shuffle=False)
        else:
            train_loader = loader
            val_loader = None

        return train_loader, val_loader
    
    
    def obtain_metrics(self, ANFISmodel, train_loader, val_loader):
        """
        Computes and records evaluation metrics for the provided ANFISmodel on training and validation sets.
        
        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param train_loader: DataLoader for training set.
        :type train_loader: torch.utils.data.DataLoader
        
        :param val_loader: DataLoader for validation set.
        :type val_loader: torch.utils.data.DataLoader

        :return loss: Loss value on the validation set (if available, else training set).
        :rtype: torch.tensor

        """
        #Validation set
        if (val_loader != None):
            x_val = val_loader.dataset.tensors[0]
            y_val = val_loader.dataset.tensors[1]

            with torch.no_grad():
                pred = ANFISmodel(x_val)

            val_loss = self.loss_function(pred, y_val.to(pred.dtype))
            self.val_history['loss'] = torch.cat([self.val_history['loss'], torch.tensor([val_loss])])

            measures = train.obtain_measures(ANFISmodel, x_val, y_val)

            for measure in measures:
                if measure not in self.val_history:
                    self.val_history[measure] = torch.tensor([])
                self.val_history[measure] =  torch.cat([self.val_history[measure], torch.tensor([measures[measure]])])

        #Training set
        x_train = train_loader.dataset.tensors[0]
        y_train = train_loader.dataset.tensors[1]

        with torch.no_grad():
            pred = ANFISmodel(x_train)

        loss = self.loss_function(pred, y_train.to(pred.dtype))
        self.history['loss'] = torch.cat([self.history['loss'], torch.tensor([loss])])

        measures = train.obtain_measures(ANFISmodel, x_train, y_train)

        for measure in measures:
            if measure not in self.history:
                self.history[measure] = torch.tensor([])
            self.history[measure] =  torch.cat([self.history[measure], torch.tensor([measures[measure]])])

        if val_loader != None:
            loss = val_loss
        return loss


    def GrowNet(self, ANFISmodel, loader):
        """
        Create a new rule in the ANFIS model based on the firing levels of each rule, the samples that model best, 
        and the number of these.

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        :returns: True if ANFISmodel is modified, False otherwise.
        :rtype: bool 

        """
        first_batch = True
        for x_batch, _ in loader:
            #Max firing levels are obtenined
            firing_levels, _, _ = ANFISmodel.intermediate_values(x_batch)
            max_fl = torch.max(firing_levels, dim=1)

            #Boolean mask to filter the samples
            dGrow_mask = (max_fl.values <= self.dGrow**ANFISmodel.input_size)

            #Necesary tensors are defined on the first iteration
            if first_batch:
                bad_samples = torch.tensor([])
                best_bs_rules = torch.tensor([], dtype=torch.int)
                first_batch = False

            #The samples are extracted by concatenating tensors (which are filtered by the mask)
            bad_samples = torch.cat((bad_samples, x_batch[dGrow_mask]), dim=0)
            best_bs_rules = torch.cat((best_bs_rules, max_fl.indices[dGrow_mask]), dim=0)

        #Ngrow parameter filter
        unique_rules, counts = torch.unique(best_bs_rules, return_counts=True)
        Ngrow_mask = (counts > self.Ngrow)

        indices_to_keep = torch.isin(best_bs_rules, unique_rules[Ngrow_mask]).nonzero().squeeze()

        bad_samples = bad_samples[indices_to_keep]
        best_bs_rules = best_bs_rules[indices_to_keep]

        #return False if ANFISmodel is not modified
        if bad_samples.size(0) == 0:
            return False

        #a list of masks called "rules" is created to calculate the necessary means and stds
        rules = [best_bs_rules == value for value in torch.unique(best_bs_rules)]

        #means and stds by rule are calculated
        means = torch.stack([(bad_samples[rule].mean(dim=0)) for rule in rules])
        stds = torch.stack([(bad_samples[rule].std(dim=0)) for rule in rules])

        #Premises and consequents modifications to add a new rule
        new_premises = torch.stack([means.t(), stds.t()], dim=2)
        ANFISmodel.set_premises(torch.cat([ANFISmodel.premises, new_premises], dim=1))

        new_consequents = 2 * torch.rand(new_premises.size(1), ANFISmodel.input_size + 1) - 1
        ANFISmodel.set_consequents(torch.cat([ANFISmodel.consequents, new_consequents]))

        #ages and freezed rules tensors updated
        self.freezed = torch.cat([self.freezed, torch.zeros(new_premises.shape[1], dtype=torch.int)])
        self.ages = torch.cat([self.ages, torch.zeros(new_premises.shape[1], dtype=torch.int)])

        #return True if ANFISmodel is modified
        return True


    def SplitSubNetwork(self, ANFISmodel, loader):
        """
        Splits a sub-network based on rule contribution and error.

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        :returns: True if ANFISmodel is modified, False otherwise.
        :rtype: bool 

        """
        first_batch = True
        for x_batch, y_batch in loader:
            #Max firing levels are obtenined
            firing_levels, _, _ = ANFISmodel.intermediate_values(x_batch)
            max_fl = torch.max(firing_levels, dim=1)

            #Necesary tensors are defined on the first iteration
            if first_batch:
                samples = torch.tensor([])
                samples_output = torch.tensor([])
                best_rules = torch.tensor([], dtype=torch.int)
                first_batch = False

            #The best rules are extracted by concatenating tensors
            samples = torch.cat((samples, x_batch), dim=0)
            samples_output = torch.cat((samples_output, y_batch), dim=0)
            best_rules = torch.cat((best_rules, max_fl.indices), dim=0)

        #Nsplit parameter filter
        unique_rules, counts = torch.unique(best_rules, return_counts=True)
        Nsplit_mask = (counts > self.Nsplit)

        indices_to_keep = torch.isin(best_rules, unique_rules[Nsplit_mask]).nonzero().squeeze()

        #return Flase if ANFISmodel is not modified
        if ((indices_to_keep.shape[0] == 0)):
            return False

        samples = samples[indices_to_keep]
        samples_output = samples_output[indices_to_keep]
        best_rules = best_rules[indices_to_keep]

        #the rules from best_rules tensor are extracted
        unique_rules = torch.unique(best_rules)

        #MSE is calculated by rule
        mse_values = torch.stack([torch.pow((samples_output[best_rules == rule] - ANFISmodel(samples[best_rules == rule])), 2).mean(dim=0) for rule in unique_rules])

        #eSplit parameter filter
        eSplit_mask = (mse_values > self.eSplit)

        #return Flase if ANFISmodel is not modified
        if ((samples.shape[0] == 0) | (unique_rules[eSplit_mask].shape[0] == 0)):
            return False

        #loop to split each rule one by one and generate the new ones
        new_premises = ANFISmodel.premises #new_premises starts being a copy of the current premises
        new_consequents = ANFISmodel.consequents #same thing with consequents
        for rule in list(torch.flip(unique_rules[eSplit_mask], [0]).long()): #the iteration is performed on a list with the rules in descending order
            #the selected premise is extracted from the new_premises tensor and placed into the to_split tensor
            new_premises = torch.cat([new_premises[:, :rule, :], new_premises[:, rule+1:, :]], dim=1)
            to_split = ANFISmodel.premises[:, rule:rule+1, :].clone()

            #the new ones are generated
            split1 = torch.cat([(to_split[:,:,0] - to_split[:,:,1]/2).unsqueeze(1), (to_split[:,:,1]/2).unsqueeze(1)], dim=2)
            split2 = torch.cat([(to_split[:,:,0] + to_split[:,:,1]/2).unsqueeze(1), (to_split[:,:,1]/2).unsqueeze(1)], dim=2)

            #both are inserted on the new premises tensor
            new_premises = torch.cat([new_premises, torch.cat([split1, split2], dim=1)], dim=1)

            #the corresponding consequent is taken away
            new_consequents = torch.cat([new_consequents[:rule, :], new_consequents[rule+1:, :]], dim=0)

            #two new consequents are added
            new_consequents = torch.cat([new_consequents, 2 * torch.rand(2, ANFISmodel.input_size + 1) - 1], dim=0)

            #same with ages and freezed rules tensor
            self.ages = torch.cat([self.ages[:rule], self.ages[rule+1:]]) #the corresponding rule is taken away
            self.freezed = torch.cat([self.freezed[:rule], self.freezed[rule+1:]])
            self.ages = torch.cat([self.ages, torch.zeros(2, dtype=torch.int)]) #the new ones are added
            self.freezed = torch.cat([self.freezed, torch.zeros(2, dtype=torch.int)])

        #after the loop, the new parameters are set
        ANFISmodel.set_premises(new_premises)
        ANFISmodel.set_consequents(new_consequents)

        #return True if ANFISmodel is modified
        return True


    def VanishNet(self, ANFISmodel, loader):
        """
        Vanishes rules from the ANFIS model based on its ages and sample counts.

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        :returns: True if ANFISmodel is modified, False otherwise.
        :rtype: bool 

        """
        first_batch = True
        for x_batch, _ in loader:
            #Max firing levels are obtenined
            firing_levels, _, _ = ANFISmodel.intermediate_values(x_batch)
            max_fl = torch.max(firing_levels, dim=1)

            #Necesary tensors are defined on the first iteration
            if first_batch:
                best_rules = torch.tensor([], dtype=torch.int)
                first_batch = False

            #the best_rules by sample are extracted
            best_rules = torch.cat((best_rules, max_fl.indices), dim=0)

        #Important info for next operations
        unique_rules, counts = torch.unique(best_rules, return_counts=True)
        all_rules = torch.arange(ANFISmodel.rules)

        #tensor with te amounts of samples modeled for each rule (including those who dont model any sample)
        total_counts = torch.zeros(ANFISmodel.rules, dtype=torch.int64)
        total_counts[unique_rules] = counts

        #if there is no changes with the modeled samples or there is no last_best_rules (first iteration of VanishNet operator)
        if torch.equal(best_rules, self.last_best_rules) | torch.equal(self.last_best_rules, torch.tensor([-1])):
            #ages update
            self.ages += 1
        else:
            #last best rules are inspected (only the ones that appeared in the current iteration)
            last_unique_rules, last_counts = torch.unique(self.last_best_rules, return_counts=True)
            last_total_counts = torch.zeros(ANFISmodel.rules, dtype=torch.int64)
            last_total_counts[last_unique_rules[last_unique_rules < ANFISmodel.rules]] = last_counts[last_unique_rules < ANFISmodel.rules]

            #rules categorization
            improved_rules = all_rules[last_total_counts < total_counts]
            not_improved_rules = all_rules[last_total_counts >= total_counts]

            #ages update
            self.ages[improved_rules] = 0
            self.ages[not_improved_rules] += 1

        #lVanish and Nvanish filters
        mask = ((self.ages > self.lVanish) & (total_counts < self.Nvanish))
        rules_to_eliminate = all_rules[mask]

        if torch.equal(rules_to_eliminate, torch.tensor([], dtype=torch.int64)):
            return False

        #Parameters update (the rules are eliminated)
        new_premises = ANFISmodel.premises
        new_consequents = ANFISmodel.consequents
        for rule in rules_to_eliminate:
            new_premises = torch.cat([new_premises[:, :rule, :], new_premises[:, rule+1:, :]], dim=1)
            new_consequents = torch.cat([new_consequents[:rule, :], new_consequents[rule+1:, :]], dim=0)
            self.ages = torch.cat([self.ages[:rule], self.ages[rule+1:]]) #the corresponding rule is taken away
            self.freezed = torch.cat([self.freezed[:rule], self.freezed[rule+1:]])

        #New parameters are set
        ANFISmodel.set_premises(new_premises)
        ANFISmodel.set_consequents(new_consequents)

        #update last best rules
        self.last_best_rules = best_rules

        #The best rules tensor is returned for next iterations, also the ages
        return True