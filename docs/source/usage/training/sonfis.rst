SONFIS
======

Self-Organizing Neuro-Fuzzy Inference System (SONFIS). Algoritmo que combina el algoritmo híbrido clásico para ANFIS con una serie de operaciones para la actualización de la estructura del modelo.

.. image:: ../../_static/SONFIS_pseudocode.png
    :width: 100%
    :align: center

.. warning::
    Este módulo solo es aplicable a modelos ANFIS homogéneos con reglas reducidas (rule_reduced h_ANFIS).

.. note::
    - Para más detalles sobre el algoritmo SONFIS, puede consultar el artículo original: `SONFIS: Structure Identification and Modeling with a Self-Organizing Neuro-Fuzzy Inference System <https://doi.org/10.1080/18756891.2016.1175809>`_.
    - Para más información sobre su implementación en el toolbox, ver :ref:`SONFIS`.
    - Puede ver ejemplos de uso :ref:`aquí <sonfis_examples>`.

Instanciación
-------------
Para instanciar el algoritmo SONFIS se deben pasar los siguientes argumentos:

- `Ngrow` (int): Número mínimo de muestras para crecer una nueva subred.
- `dGrow` (float): Si el nivel de disparo máximo de un conjunto de muestras mal modeladas es menor o igual a este valor elevado a la dimensión de los datos, se considera para crecer una nueva subred.
- `Nsplit` (int): Número mínimo de muestras para dividir una subred.
- `eSplit` (float): Error cuadrado medio mínimo de un conjunto de muestras para dividir una subred.
- `Nvanish` (int): Número máximo de muestras para desvanecer una subred.
- `lVanish` (int): Edad máxima para desvanecer una subred.
- `max_iterations` (int): Número máximo de iteraciones.
- `ANFIStrainer` (Hybrid_learning_algorithm): Instancia del algoritmo de aprendizaje híbrido. Se utiliza para obtener sus parámetros, ya que el algoritmo de actualización de parámetros híbrido del modelo ANFIS clásico es por defecto el aplicado en SONFIS.
- `validation` (float): Porcentaje de división de validación. Si es 0, no se realiza validación (Default: 0).
- `early_stopping` (EarlyStopping): Instancia de EarlyStopping (Default: None).
- `last_training_iteration` (bool): Indica si se debe realizar una última actualización de parámetros después de que el algoritmo SONFIS finalice (Default: False).

Llamado
-------
Para llamar al algoritmo SONFIS, se debe pasar el modelo a entrenar y el DataLoader con los datos de entrenamiento. Adicionalmente, se puede especificar si se desea mostrar información del progreso del entrenamiento con el argumento `verbose`.

Ejemplo
#######
Para instanciar el algoritmo, primero hay que instanciar el algoritmo de entrenamiento híbrido:

.. code:: python

    import neuro_fuzzy_toolbox as nft

    epochs = 500
    loss_fn = nn.functional.mse_loss
    optimizer = torch.optim.AdamW
    params = {'lr': 0.0001, 'weight_decay': 0.001}
    early_stopping = nft.EarlyStopping(patience=40, delta=0.0001)

    ANFIStrainer = nft.Hybrid_learning_algorithm(
        epochs=epochs,
        loss_function=loss_fn,
        optimizer=optimizer,
        optimizer_params=params,
        early_stopping=early_stopping
    )

.. important::
    El argumento `validation` para la instancia del algoritmo de entrenamiento híbrido no es tomado en cuenta, pues este se define en la instancia del algoritmo SONFIS.
    Es decir, independientemente de si se define o no en el algoritmo de entrenamiento híbrido, se debe definir en el algoritmo SONFIS (el mecanismo early stopping
    del algoritmo de entrenamiento híbrido trabajará con el conjunto de validación definido en SONFIS).

Luego, se puede instanciar el algoritmo SONFIS:

.. code:: python

    Ngrow = 30
    dGrow = 0.8
    Nsplit = 25
    eSplit = 1.2
    Nvanish = 5
    lVanish = 3
    max_iterations = 100
    validation = 0.3
    sonfis_early_stopping = nft.EarlyStopping(patience=7, delta=0.01)
    last_training_iteration = True

    sonfis = nft.SONFIS(
        Ngrow=Ngrow,
        dGrow=dGrow,
        Nsplit=Nsplit,
        eSplit=eSplit,
        Nvanish=Nvanish,
        lVanish=lVanish,
        max_iterations=max_iterations,
        ANFIStrainer=ANFIStrainer,
        validation=validation,
        early_stopping=sonfis_early_stopping,
        last_training_iteration=last_training_iteration
    )

Finalmente, se puede proceder con el entrenamiento del modelo:

.. code:: python

    # Considerando que "model" es el modelo ANFIS instanciado en cuestión
    # y que "train_loader" es el DataLoader con los datos de entrenamiento

    sonfis(model, train_loader, verbose=True) # Esto ejecutará el entrenamiento del modelo





