.. _sonfis_examples:

SONFIS Examples
===============

Example 1:
----------

0. Imports
##########
Importamos las librerías necesarias para el ejemplo:

.. code:: python

    import sys
    import os
    import torch
    import torch.utils.data as data
    import torch.nn as nn

    sys.path.append(os.path.abspath('../../'))

    import neuro_fuzzy_toolbox as nft

    import numpy as np

    from sklearn.preprocessing import (
        MinMaxScaler,
        StandardScaler
    )
    from sklearn.model_selection import train_test_split

El ejemplo consiste en un dataset de UC Irvine Machine Learning Repository llamado "Parkinsons" con 22 features y 1 target, para más detalles sobre el dataset puede consultar `aquí <https://archive.ics.uci.edu/dataset/174/parkinsons>`_.

.. code:: python

    from ucimlrepo import fetch_ucirepo

    parkinsons = fetch_ucirepo(id=174)

    X = parkinsons.data.features
    y = parkinsons.data.targets

1. Definición de los datos
##########################
El 30% de los datos se usarán para testeo y el 70% restante para entrenamiento:

.. code:: python

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

2. Preprocesamiento de los datos
################################

.. code:: python

    scaler = MinMaxScaler(feature_range=(-1, 1))

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

3. Definición del DataLoader
############################
Los conjuntos de datos se transforman a tensores de pytorch y se define el DataLoader para el conjunto de entrenamiento junto con el tamaño de los batches en las iteraciones del posterior entrenamiento del modelo.

.. code:: python

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train.values).squeeze()
    y_test = torch.from_numpy(y_test.values).squeeze()

    loader = data.DataLoader(
        data.TensorDataset(
            x_train, 
            y_train), 
        batch_size = 8, 
        shuffle = True
    )

    x_train = loader.dataset.tensors[0]
    y_train = loader.dataset.tensors[1]

4. Definición del modelo
########################
SONFIS solo trabaja con modelos ANFIS rule reduced.

.. code:: python

    model = nft.rule_reduced_ANFIS(
        input_size = x_train.shape[1],
        num_mfs = 1,
        outputs = 2,
        output_type='multiclass',
        dtype=torch.float64
    )

    model.init_premises(x_train)

5. Definición del algoritmo de entrenamiento
############################################
Luego, podemos definir el algoritmo de entrenamiento junto con un mecanismo **EarlyStopping** implementado en el toolbox:

.. note::
    Recordar que el parámtetro **validation** del algoritmo de actualización de parámetros no se toma en cuenta cuando se aplica al modelo SONFIS, pues
    este se indica al momento de instanciar el algoritmo SONFIS como tal. Si se instancia el algoritmo de actualización de parámetros con el parámetro **validation**,
    este no se tomará en cuenta de todas maneras.

.. code:: python

    loss_fn = nn.functional.cross_entropy
    epochs = 1000
    optimizer = torch.optim.Adam
    params = {'lr': 0.0001}

    early_stopping = nft.EarlyStopping(
        patience=30, 
        delta=0.0001
    )

    trainer = nft.Hybrid_learning_algorithm(
        epochs=epochs,
        loss_function=loss_fn,
        optimizer=optimizer,
        optimizer_params=params,
        early_stopping=early_stopping
    )

6. Instancición del algoritmo SONFIS
####################################
Se instancia el algoritmo SONFIS con el modelo ANFIS previamente definido y el algoritmo de entrenamiento:

.. code:: python

    Ngrow = 20
    dGrow = 0.8
    Nsplit = 20
    eSplit = 0.7
    Nvanish = 5
    lVanish = 2

    max_iterations = 40

    anfis_trainer = trainer

    validation = 0.2
    sonfis_early_stopping = nft.EarlyStopping(patience=5, delta=0.01)
    last_training_iteration = True


    sonfis = nft.SONFIS(
        Ngrow=Ngrow,
        dGrow=dGrow,
        Nsplit=Nsplit,
        eSplit=eSplit,
        Nvanish=Nvanish,
        lVanish=lVanish,
        max_iterations=max_iterations,
        ANFIStrainer=anfis_trainer,
        validation=validation,
        early_stopping=sonfis_early_stopping,
        last_training_iteration=last_training_iteration
    )

7. Entrenamiento del modelo
###########################

.. code:: python

    sonfis(model, loader, verbose=True)

.. code:: python

    Iteration:  0/40 - loss: 1.613511 - validation loss: 1.767238
     -> ANFIS rules: 1

    Iteration:  1/40 - loss: 0.446700 - validation loss: 0.470952
     -> ANFIS rules: 2

    Iteration:  2/40 - loss: 0.427127 - validation loss: 0.421758
     -> ANFIS rules: 4

    No more updates
    Iteration:  3/40 - loss: 0.427127 - validation loss: 0.421758
     -> ANFIS rules: 4


    Training finished
     -> ANFIS rules: 4


7. Evaluación del modelo
########################
Finalmente, podemos evaluar el modelo usando el conjunto de testeo y la función **get_measures** del toolbox si se desea:

.. code:: python

    test_measures = nft.get_measures(model, x_test, y_test)

    for measure in test_measures:
        print(measure + ':', test_measures[measure])

.. code:: python

    Accuracy: 0.847457627118644
    Precision: 0.8412369275153264
    Recall: 0.847457627118644
    F1: 0.8414974855652821
    Confusion Matrix: [[ 9  6]
     [ 3 41]]