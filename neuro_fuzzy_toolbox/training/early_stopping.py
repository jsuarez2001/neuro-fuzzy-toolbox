class EarlyStopping():
    def __init__(self, patience, delta=0, last_state=False):
        #Parameters
        self.patience = patience
        self.delta = delta
        self.last_state = last_state # True if the last state is the one to be restored (not the best one)
        
        #For running
        self._counter = 0
        self._best_loss = None
        self._best_premises = None
        self._best_consequents = None
        self._stop = False

    def __call__(self, ANFISmodel, loss, verbose=False):
        if self._best_loss is None:
            self._best_loss = loss
            self._best_premises = ANFISmodel.get_premises()
            self._best_consequents = ANFISmodel.get_consequents()

        elif loss + self.delta > self._best_loss:
            self._counter += 1
            if self._counter >= self.patience:
                self._stop = True
                if verbose:
                    print('Early stopping')
                if self.last_state == False:
                    ANFISmodel.set_premises(self._best_premises)
                    ANFISmodel.set_consequents(self._best_consequents)

        else:
            self._best_loss = loss
            self._best_premises = ANFISmodel.get_premises()
            self._best_consequents = ANFISmodel.get_consequents()
            self._counter = 0

    def reset(self):
        self._counter = 0
        self._best_loss = None
        self._best_premises = None
        self._best_consequents = None
        self._stop = False