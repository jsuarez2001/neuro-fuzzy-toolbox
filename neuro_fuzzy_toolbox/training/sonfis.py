import torch

from neuro_fuzzy_toolbox.training import Hybrid_learning_algorithm

class SONFIS(Hybrid_learning_algorithm):
    def __init__(self, Ngrow, dGrow, Nsplit, eSplit, Nvanish, lVanish, max_iterations, ANFIS_trainer, validation=0, early_stopping=None, last_training_iteration=False):
        
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
        self.epochs = ANFIS_trainer.epochs
        self.loss_function = ANFIS_trainer.loss_function
        
        # Optimizer
        self.optimizer = ANFIS_trainer.optimizer
        self.optimizer_params = ANFIS_trainer.optimizer_params
        
        # Early stopping
        self.trainer_validation = ANFIS_trainer.validation
        self.trainer_early_stopping = ANFIS_trainer.early_stopping
        
        
        # ------ Internal variables ------
        self.freezed = torch.tensor([], dtype=torch.int)
        self.ages = torch.tensor([], dtype=torch.int)
        self.last_best_rules = torch.tensor([-1], dtype=torch.int)
        

    def __call__(self, ANFISmodel, loader, verbose=True):
        train_loader, val_loader = self._train_val_split(loader)
        self._register_loss(ANFISmodel, train_loader, val_loader)
        
        self.ages = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int)
        self.freezed = torch.zeros(ANFISmodel.fuzzy_rules, dtype=torch.int)
        
        model_updated = True
        i = 0
        
        self.call_ANFIStrainer(ANFISmodel, loader)
        while(model_updated and i < self.max_iterations):
            did_Grow = False
            did_Split = False
            did_Vanish = False
            
            if verbose:
                print(f'Iteration: {i+1}/{self.max_iterations}')
            
            self.freeze_rules()
            
            # Grow
            did_Grow = self.GrowNet(ANFISmodel, loader)
            
            # Split
            if not did_Grow:
                did_Split = self.SplitNet(ANFISmodel, loader)
                
            # Vanish
            did_Vanish = self.VanishNet(ANFISmodel, loader)
            
            if verbose:
                print(f'\n  Fuzzy rules: {ANFISmodel.fuzzy_rules}')
            
            model_updated = did_Grow or did_Split or did_Vanish
            
            if model_updated:
                self.call_ANFIStrainer(ANFISmodel, loader)
            else:
                if verbose:
                    print('No more updates')
                    
            self._register_loss(ANFISmodel, train_loader, val_loader)
            
            if self.validation > 0 and self._check_early_stop(ANFISmodel, self.val_history["loss"][-1], verbose):
                break
            
            if verbose:
                print('loss:', self.history["loss"][-1])
                if self.validation > 0:
                    print('val_loss:', self.val_history["loss"][-1])
            
            i += 1
            
        self.unfreeze_rules()
        
        if self.last_training_iteration:
            self.call_ANFIStrainer(ANFISmodel, loader)
            
        self._register_loss(ANFISmodel, train_loader, val_loader)
            
    
    def freeze_rules(self):
        self.freezed = torch.ones_like(self.freezed)
        
    def unfreeze_rules(self):
        self.freezed = torch.zeros_like(self.freezed)
        
    def call_ANFIStrainer(self, ANFISmodel, loader):
        pass
    
    def _check_early_stop(self, ANFISmodel, loss, verbose):
        if self.early_stopping is not None:
            self.early_stopping(ANFISmodel, loss, verbose)
            if self.early_stopping._stop:
                self.freezed = torch.zeros_like(self.freezed)
                self.ages = self.ages[:ANFISmodel]
                return True
        return False
        
    def GrowNet(self, ANFISmodel, loader):
        pass
    
    def SplitNet(self, ANFISmodel, loader):
        pass
    
    def VanishNet(self, ANFISmodel, loader):
        pass
