import torch
import torch.nn as nn

import anfis.train as train


class SONFIS():
    '''
    Self-Organizing Algorithm for Neuro-Fuzzy Inference Systems (SONFIS) for online learning and adaptation.

    **Attributes:**
    
    .. attribute:: max
    
        Maximum number of iterations for SONFIS learning.
        
        :type: int
    
    .. attribute:: y
    
        Learning rate for OLS (Ordinary Least Squares) optimizer.
        
        :type: float
        
    .. attribute:: OLSepochs
    
        Number of epochs for OLS optimizer during each SONFIS iteration.
        
        :type: int
    
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
        
    .. attribute:: loss_function
    
        Loss function for training (specifically, to update the premise parameters).
        
        :type: torch.nn.Module
        :default: nn.functional.mse_loss
        
    .. attribute:: validation
    
        Proportion of the training data to use for validation.
        
        :type: float
        :default: 0
        
    .. attribute:: ols
    
        The OLS instance to be applied while the SONFIS algorithm is running.

        :type: OLS
        
    .. attribute:: freezed
    
        Tensor indicating which rules are frozen (whose parameters should not be updated).

        :type: torch.tensor
        
    .. attribute:: ages
    
        Tensor that contains the ages of each of the rules of the ANFIS model.

        :type: torch.tensor
        
    .. attribute:: last_best_rules

        Tensor containing which rules modeled the most samples in the last iteration (used in the VanishNet operator).
        
        :type: torch.tensor
        
    .. attribute:: train_history

        Training loss history.
        
        :type: torch.tensor
        
    .. attribute:: val_history
    
        Validation loss history.
        
        :type: torch.Tensor
    
    **To initialize it:**
    
    The parameters that must be taken into account are the following:
    
    :param max: Maximum number of iterations for SONFIS learning.
    :type max: int
    
    :param y: Learning rate for premises update.
    :type y: float
    
    :param OLSepochs: Number of training epochs.
    :type OLSepochs: int
    
    :param Ngrow: Threshold for growing a new rule based on firing levels.
    :type Ngrow: int
    
    :param dGrow: Threshold for deciding whether to grow a new rule based on firing level spread.
    :type dGrow: float
    
    :param Nsplit: Threshold for splitting a sub-network based on rule contribution.
    :type Nsplit: int
    
    :param eSplit: Threshold for deciding whether to split a sub-network based on rule error.
    :type eSplit: float
    
    :param Nvanish: Threshold for deciding whether to vanish a rule based on the number of samples.
    :type Nvanish: int
    
    :param lVanish: Threshold for deciding whether to vanish a rule based on its age.
    :type lVanish: int
        
    :param loss_function: Loss function for training (default: nn.functional.mse_loss).
    :type loss_function: torch.nn.Module
        
    :param validation: Proportion of the training data to use for validation (default: 0).
    :type validation: float

    **Example Usage:**
    
    .. code::
    
        >>> train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        >>> x_train = train_loader.dataset.tensors[0]
        >>> anfis_model = Type3ANFIS(x_train, init_rules=1)
        >>> sonfis = SONFIS(max=50, y=0.01, OLSepochs=5, Ngrow=20, dGrow=1.5, Nsplit=30, eSplit=0.2, Nvanish=2, lVanish=15)
        >>> sonfis(anfis_model, train_loader)
        >>> print(sonfis.train_history)
    
    **Methods:**

    '''
    def __init__(self, max, y, OLSepochs, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, loss_function=nn.functional.mse_loss, validation=0):
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
        self.y = y
        self.OLSepochs = OLSepochs
        self.Ngrow = Ngrow
        self.dGrow = dGrow
        self.Nsplit = Nsplit
        self.eSplit = eSplit
        self.Nvanish = Nvanish
        self.lVanish = lVanish
        self.validation = validation

        #OLS
        self.ols = train.OLS(OLSepochs, y, loss_function, validation)

        #Other attributes
        self.freezed = torch.tensor([], dtype=torch.int)
        self.ages = torch.tensor([], dtype=torch.int)
        self.last_best_rules = torch.tensor([-1], dtype=torch.int)

        #history
        self.train_history = torch.tensor([])
        self.val_history = torch.tensor([])


    def __call__(self, ANFISmodel, loader):
        """
        Performs SONFIS learning on the provided ANFIS model and data loader.
        
        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        """
        self.ages = torch.zeros(ANFISmodel.rules)
        self.freezed = torch.zeros(ANFISmodel.rules)

        model_updated = True
        i = 0

        self.callOLS(ANFISmodel, loader)
        while(model_updated & (i < self.max)):
            print("\n *******ITERATION:", i+1, " ******* ")
            self.freezed = torch.ones(ANFISmodel.rules)

            #GrowNet
            did_Grow = self.GrowNet(ANFISmodel, loader)

            #Split Sub-network
            if not did_Grow:
                did_Split = self.SplitSubNetwork(ANFISmodel, loader)
            else:
                did_Split = False

            #VanishNet
            did_Vanish = self.VanishNet(ANFISmodel, loader)

            print("\nRules amount:", ANFISmodel.rules, "\n\n")

            model_updated = did_Grow | did_Split | did_Vanish
            if model_updated:
                self.callOLS(ANFISmodel, loader)

            i += 1

        self.freezed = torch.zeros(ANFISmodel.rules)
        self.callOLS(ANFISmodel, loader)


    def callOLS(self, ANFISmodel, loader):
        """
        Calls the OLS optimizer to update the ANFIS model and store the loss history.

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader

        """
        self.ols(ANFISmodel, loader, self.freezed)
        self.train_history = torch.cat([self.train_history, self.ols.train_history])


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
        self.freezed = torch.cat([self.freezed, torch.zeros(new_premises.shape[1])])
        self.ages = torch.cat([self.ages, torch.zeros(new_premises.shape[1], dtype=torch.bool)])

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
            self.ages = torch.cat([self.ages, torch.zeros(2, dtype=torch.bool)]) #the new ones are added
            self.freezed = torch.cat([self.freezed, torch.zeros(2)])

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