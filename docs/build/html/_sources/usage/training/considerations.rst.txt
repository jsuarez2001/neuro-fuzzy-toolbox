Consideraciones Iniciales
==========================

Entrada y salida de los modelos
-------------------------------
Los modelos presentados en Neuro-Fuzzy Toolbox están diseñados para trabajar manteniendo la siguiente lógica:

Para las entradas, se espera que los datos mantengan su dimensionalidad de esta manera:

- Para 1 feature: (batch_size, )
- Para 2 o más features: (batch_size, n_features)

Y con respecto a las salidas:

- Para una sola salida: (batch_size, )
- Para múltiples salidas: (batch_size, n_outputs)

.. note::
    El concepto de múltiple salida, además de referirse a más de una salida como tal, también se refiere al concepto de la salida multiclase, es decir, n_outputs puede entenderse también como la cantidad de clases en un problema de clasificación.

Esto es importante tenerlo en cuenta al momento de definir los DataLoaders para el entrenamiento de los modelos y para su posterior evaluación.

.. _DataLoaders_usage:

DataLoaders de PyTorch
----------------------
Los algoritmos de entrenamiento disponibles en Neuro-Fuzzy Toolbox trabajan con DataLoaders de PyTorch. Estos objetos permiten cargar los datos de entrenamiento y validación de manera eficiente y sencilla, por lo que
muchas de los mecanismos internos de los algoritmos de entrenamiento están diseñados para trabajar con estos objetos y reciben estas estructuras de datos como argumentos para su ejecución.

A continuación, se presenta un simple ejemplo de cómo definir un DataLoader para un conjunto de datos de entrenamiento, pero para información más detallada se recomienda revisar la documentación oficial de PyTorch:  
`Datasets & DataLoaders <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#>`_

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Simulamos un conjunto de datos de entrenamiento con 200 muestras y 3 features
    x_train = torch.rand(200, 3)
    y_train = torch.rand(200, )

Al definir un DataLoader, se debe tener en cuenta que el primer argumento debe ser un objeto de tipo TensorDataset, el cual recibe los datos de entrada y salida.
Debemos también definir el batch_size, pues es con este parámetro que se definirá la cantidad de muestras que se utilizarán en cada iteración del entrenamiento.
El parámtro shuffle=True permite que los datos se mezclen en cada iteración, esto es opcional.

.. code-block:: python

    dataset = TensorDataset(x_train, y_train)

    # Definimos un DataLoader con un batch_size de 32 y shuffle=True
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

Validación y Test
-----------------
- Con respecto al típico split de los datos en entrenamiento y validación, esto se realiza internamente en los algoritmos de entrenamiento de Neuro-Fuzzy Toolbox controlando un parámetro.
- Para el conjunto de datos de test, queda en manos del usuario dejar un conjunto de datos de testeo aparte (antes de definir el DataLoader de entrenamiento) para evaluar el modelo una vez entrenado.

Algortimos de entrenamiento
---------------------------
Los algoritmos de entrenamiento disponibles en Neuro-Fuzzy Toolbox son:

- **Hybrid Learning Algorithm**
- **Basic Optimizer Training Algorithm**
- **Double Optimizer Training Algorithm**
- **SONFIS**

y para el uso de cada uno de ellos solo se deben tener en cuenta 2 métodos: 

- **Inicialización** (*__init__*): Para crear las instancias de los modelos (definiendo los parámetros y/u otros mecanismos).
- **Entrenamiento** (*__call__*): Para entrenar los modelos con los DataLoaders definidos.