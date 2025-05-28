class EarlyStopping():
    """
    Mecanismo Early Stopping para detener el entrenamiento de un modelo de aprendizaje automático (torch.nn.Module).
    """
    def __init__(self, patience, delta=0, last_state=False):
        """
        Inicializa una nueva instancia de la clase EarlyStopping.
        
        Args:
            patience (int): Número de épocas sin mejora antes de detener el entrenamiento.
            delta (float): Valor mínimo de mejora para considerar que hubo mejora en el entrenamiento (Default: 0).
            last_state (bool): Indica si se debe restaurar el último estado del modelo o el mejor estado encontrado (Default: False).

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
        Llama al mecanismo de Early Stopping para evaluar si se debe detener el entrenamiento del modelo. En caso de que así sea, el atributo stop se actualiza a True.
        
        Args:
            model (torch.nn.Module): Modelo a evaluar
            loss (float): Pérdida actual del modelo.
            verbose (bool): Indica si se deben imprimir mensajes de aviso (Default: False).
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
        Reinicia el mecanismo de Early Stopping.
        """
        self._counter = 0
        self._best_loss = None
        self._best_state_dict = None
        self._stop = False
        
    @property
    def stop(self):
        """
        Booleano que indica si el entrenamiento debe detenerse.
        """
        return self._stop