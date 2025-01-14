class EarlyStopping():
    def __init__(self, patience, delta=0, last_state=False):
        """
        Initializes an Early Stopping mechanism to monitor the ANFIS training progress.

        Parameters:
        - patience (int): Number of epochs with no improvement after which training will be stopped.
        - delta (float): Minimum change in the monitored quantity to qualify as an improvement (default: 0).

        """
        #Parameters
        self.patience = patience
        self.delta = delta
        self.last_state = last_state

        #For running
        self.counter = 0
        self.best_loss = None
        self.best_premises = None
        self.best_consequents = None
        self.early_stop = False

    def __call__(self, loss, ANFISmodel):
        """
        Monitors the loss during training and determines whether to stop early.

        Parameters:
        - loss (float): Current loss value.
        - ANFISmodel (ANFIS): Instance of the ANFIS model being trained.

        """
        if self.best_loss is None:
            self.best_loss = loss
            self.best_premises = ANFISmodel.premises
            self.best_consequents = ANFISmodel.consequents

        elif loss + self.delta > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.last_state == False:
                    ANFISmodel.set_premises(self.best_premises)
                    ANFISmodel.set_consequents(self.best_consequents)

        else:
            self.best_loss = loss
            self.best_premises = ANFISmodel.premises
            self.best_consequents = ANFISmodel.consequents
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.best_premises = None
        self.best_consequents = None
        self.early_stop = False