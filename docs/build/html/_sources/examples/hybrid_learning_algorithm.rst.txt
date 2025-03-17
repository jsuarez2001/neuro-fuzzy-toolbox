.. _hybrid_learning_algorithm_examples:

Hybrid Learning Algorithm Examples
==================================

Example 1: Surface Approximation (h_ANFIS)
------------------------------------------

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

Para este ejemplo en particular, se intentará aproximar una superficie en el espacio R3:

.. math::

    f(x, y) = 3\cdot (1-x)^2\cdot e^{-x^2 - (y+1)^2} - 10\cdot (\frac{1}{5}x - x^3 - y^5)\cdot e^{-x^2-y^2} - \frac{1}{3}\cdot e^{-(x+1)^2 - y^2}

1. Definición de los datos
##########################
Conjunto entrenamiento (simulando cierto nivel de ruido) y testeo:

.. code:: python

    def z(x, y):
        return ((3) * ((1-x)**2) * (np.exp(-(x**2)-((y+1)**2))) - (10) * ((x/5)-(x**3)-(y**5)) * (np.exp(-(x**2)-(y**2))) - (1/3)*np.exp(-(x+1)**2-(y**2)))

    #Training
    x0 = np.random.uniform(-3,3,1000)
    x1 = np.random.uniform(-3,3,1000)

    e = np.random.normal(0,1.0,1000) #noise
    Y = z(x0,x1) + e

    #Testing
    x0_test = np.random.uniform(-3,3,1000)
    x1_test = np.random.uniform(-3,3,1000)

    Y_test = z(x0_test,x1_test)

2. Preprocesamiento de los datos
################################
Escalado de los datos:

.. code:: python

    #Training
    scaler = MinMaxScaler(feature_range=(-1, 1))
    vstack_train = np.vstack((x0,x1)).T
    scaled_train = scaler.fit_transform(vstack_train)

    #Testing
    vstack_test = np.vstack((x0_test,x1_test)).T
    scaled_test = scaler.transform(vstack_test)

3. Definición del DataLoader
############################
Los conjuntos de datos se transforman a tensores de pytorch y se define el DataLoader para el conjunto de entrenamiento junto con el tamaño de los batches en las iteraciones del posterior entrenamiento del modelo.

.. code:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_loader = data.DataLoader(
        data.TensorDataset(
            torch.tensor(scaled_train, dtype=torch.float32), 
            torch.tensor(Y, dtype=torch.float32)
            ), 
        batch_size = 32, 
        shuffle = True)

    # Extraemos los tensores para su uso posterior si es necesario
    x_train = train_loader.dataset.tensors[0]
    y_train = train_loader.dataset.tensors[1]

    x_test = torch.tensor(scaled_test, dtype=torch.float32)
    y_test = torch.tensor(Y_test, dtype=torch.float32)

4. Definición del modelo
########################
En este caso, se optará por el uso de un modelo h_ANFIS (ANFIS homogéneo).

.. note:: 
    El Hybrid_learning_algorithm es aplicable a cualquier modelo ANFIS disponible en Neuro-Fuzzy Toolbox (ANFIS y h_ANFIS).

.. code:: python

    model = nft.h_ANFIS(
        input_size = 2,
        num_mfs = 6,
    )

Inicializar los parámetros de las funciones de membresía tiene mejores resultados en este ejemplo:

.. code:: python

    model.init_premises(x_train)

    model.show_premises_structure()

.. code::

          a (x0)  b (x0)  c (x0)  a (x1)  b (x1)  c (x1)
    MF 0     0.2     4.0    -1.0     0.2     4.0    -1.0
    MF 1     0.2     4.0    -0.6     0.2     4.0    -0.6
    MF 2     0.2     4.0    -0.2     0.2     4.0    -0.2
    MF 3     0.2     4.0     0.2     0.2     4.0     0.2
    MF 4     0.2     4.0     0.6     0.2     4.0     0.6
    MF 5     0.2     4.0     1.0     0.2     4.0     1.0


5. Definición del algoritmo de entrenamiento
############################################
Luego, podemos definir el algoritmo de entrenamiento junto con un mecanismo **EarlyStopping** implementado en el toolbox:

.. code:: python

    epochs = 500
    loss_fn = nn.functional.mse_loss
    optimizer = torch.optim.AdamW
    params = {'lr': 0.0001, 'weight_decay': 0.001}
    validation = 0.3
    early_stopping = nft.EarlyStopping(patience=30)

    trainer = nft.Hybrid_learning_algorithm(
        epochs=epochs,
        loss_function=loss_fn,
        optimizer=optimizer,
        optimizer_params=params,
        validation=validation,
        early_stopping=early_stopping
    )

6. Entrenamiento del modelo
###########################

.. code:: python

    trainer(model, train_loader, verbose=True)

.. code:: python

    Epoch:   1/500 - loss: 0.480801 - validation loss: 0.721475
    Epoch:   2/500 - loss: 0.477840 - validation loss: 0.718508
    Epoch:   3/500 - loss: 0.475277 - validation loss: 0.716161
    Epoch:   4/500 - loss: 0.472659 - validation loss: 0.713498
    Epoch:   5/500 - loss: 0.470017 - validation loss: 0.711177
    Epoch:   6/500 - loss: 0.467647 - validation loss: 0.708426
    Epoch:   7/500 - loss: 0.464859 - validation loss: 0.706709
    Epoch:   8/500 - loss: 0.462323 - validation loss: 0.703486
    Epoch:   9/500 - loss: 0.460128 - validation loss: 0.701562
    Epoch:  10/500 - loss: 0.457874 - validation loss: 0.698973
    Epoch:  11/500 - loss: 0.455350 - validation loss: 0.696706
    Epoch:  12/500 - loss: 0.452771 - validation loss: 0.694189
    Epoch:  13/500 - loss: 0.450527 - validation loss: 0.692190
    Epoch:  14/500 - loss: 0.448317 - validation loss: 0.689814
    Epoch:  15/500 - loss: 0.446008 - validation loss: 0.687502
    Epoch:  16/500 - loss: 0.443802 - validation loss: 0.684893
    Epoch:  17/500 - loss: 0.441731 - validation loss: 0.683295
    Epoch:  18/500 - loss: 0.439670 - validation loss: 0.680978
    Epoch:  19/500 - loss: 0.437192 - validation loss: 0.678953
    Epoch:  20/500 - loss: 0.435192 - validation loss: 0.676556
    Epoch:  21/500 - loss: 0.433183 - validation loss: 0.674573
    Epoch:  22/500 - loss: 0.431081 - validation loss: 0.672719
    Epoch:  23/500 - loss: 0.429033 - validation loss: 0.670726
    Epoch:  24/500 - loss: 0.427356 - validation loss: 0.669143
    Epoch:  25/500 - loss: 0.425483 - validation loss: 0.667172
    ...
    Epoch: 306/500 - loss: 0.375460 - validation loss: 0.625092
    Epoch: 307/500 - loss: 0.375455 - validation loss: 0.624849
    Early stopping
    Training finished


7. Evaluación del modelo
########################
Finalmente, podemos evaluar el modelo usando el conjunto de testeo y la función **get_measures** del toolbox si se desea:

.. code:: python

    test_measures = nft.get_measures(model, x_test, y_test)

    for measure in test_measures:
        print(measure + ':', test_measures[measure])

.. code::

    MSE: 0.12230610102415085
    RMSE: 0.34972289204597473
    MAE: 0.2811751365661621
    R2: 0.9658823609352112
    MAPE: 153.28375244140625
