class EarlyStopping():
    """
    Early stopping mechanism for halting the training of a machine learning model (``torch.nn.Module``) when no 
    sufficient improvement is observed.
    """
    def __init__(self, patience, delta=0, last_state=False):
        """
        Initializes a new EarlyStopping instance.
        
        Args:
            patience (int): Number of epochs without improvement before stopping training.
            delta (float): Minimum improvement required to consider that the model has improved. Defaults to ``0``.
            last_state (bool): If ``True``, restores the last model state when stopping instead of the best state 
                found during training. Defaults to ``False``.
        """
        #Parameters
        self.patience = patience
        self.delta = delta
        self.last_state = last_state # True if the last state is the one to be restored (not the best one)
        
        #For running
        self._counter = 0
        self._best_loss = None
        self._best_state_dict = None
        self._stop = False

    def __call__(self, model, loss, verbose=False):
        """
        Evaluates whether training should be stopped. If the stopping criterion
        is met, the ``stop`` attribute is updated to ``True``.
        
        Args:
            model (torch.nn.Module): Model to evaluate.
            loss (float): Current loss value of the model.
            verbose (bool): If ``True``, prints a warning message when early stopping is triggered. Defaults to ``False``.
        """
        if self._best_loss is None:
            self._best_loss = loss
            self._best_state_dict = model.state_dict()

        elif loss + self.delta > self._best_loss:
            self._counter += 1
            if self._counter >= self.patience:
                self._stop = True
                if verbose:
                    print('\nEARLY STOPPING')
                if self.last_state == False:
                    model.load_state_dict(self._best_state_dict)

        else:
            self._best_loss = loss
            self._best_state_dict = model.state_dict()
            self._counter = 0

    def reset(self):
        """
        Resets the early stopping mechanism to its initial state.
        """
        self._counter = 0
        self._best_loss = None
        self._best_state_dict = None
        self._stop = False
        
    @property
    def stop(self):
        """
        Indicates whether training should be stopped.
        
        Returns:
            bool: ``True`` if the stopping criterion has been met, ``False`` otherwise.
        """
        return self._stop