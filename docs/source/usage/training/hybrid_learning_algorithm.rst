Hybrid Learning Algorithm
=========================
Algoritmo híbrido de aprendizaje clásico para los modelos ANFIS. Combina la estimación por mínimos cuadrados para los parámetros consecuentes con un mecanismo 
de entrenamiento basado en optimizadores para los parámetros de las funciones de membresía. 

.. image:: ../../_static/Hybrid_learning_algorithm_pseudocode.png
    :width: 100%
    :align: center

.. important::
    Es aplicable a todos los modelos ANFIS disponibles en Neuro-Fuzzy Toolbox (modelos que hereden de la base_ANFIS, es decir, ANFIS y h_ANFIS).

.. note::
    - Este algoritmo es una implementación de una variante del algoritmo de aprendizaje híbrido propuesto por Jang en 1993. Para más información, se recomienda revisar el artículo original: `ANFIS: Adaptive-Network-Based Fuzzy Inference System <https://doi.org/10.1109/21.256541>`_.
    - Para más detalles sobre su implementación en el toolbox, ver :ref:`Hybrid Learning Algorithm`.
    - Puede ver ejemplos de uso :ref:`aquí <hybrid_learning_algorithm_examples>`.

Instanciación
-------------
Para instanciar el algoritmo de entrenamiento basado en optimizadores, se deben pasar los siguientes argumentos:

- `epochs` (int): Número de épocas de entrenamiento.
- `loss_function` (torch.nn.functional): Función de pérdida a utilizar durante el entrenamiento.
- `validation` (float): Proporción de los datos de entrenamiento a utilizar como conjunto de validación (opcional).
- `early_stopping` (nft.EarlyStopping): Mecanismo de parada temprana a utilizar durante el entrenamiento (opcional).
- `optimizer` (torch.optim.Optimizer): Optimizador a utilizar durante el entrenamiento.
- `optimizer_params` (dict): Parámetros adicionales a pasar al optimizador (opcional).

Llamado
-------
Para llamar al algoritmo de entrenamiento, se debe pasar el modelo a entrenar y el DataLoader con los datos de entrenamiento. 
Adicionalmente, se puede especificar si se desea mostrar información del progreso del entrenamiento con el argumento `verbose`.

Ejemplo
#######
Para instanciar el algoritmo:

.. code:: python

    import neuro_fuzzy_toolbox as nft

    epochs = 500
    loss_fn = nn.functional.mse_loss
    optimizer = torch.optim.AdamW
    params = {'lr': 0.0001, 'weight_decay': 0.001}
    validation = 0.3
    early_stopping = nft.EarlyStopping(patience=30, delta=0.001)

    trainer = nft.Hybrid_learning_algorithm(
        epochs=epochs,
        loss_function=loss_fn,
        optimizer=optimizer,
        optimizer_params=params,
        validation=validation,
        early_stopping=early_stopping
    )

Ahora, asumiendo que ya se tiene un modelo ANFIS instanciado y que se cuenta con un DataLoader de Pytorch con los datos de entrenamiento, se puede proceder con el entrenamiento del modelo:

.. code:: python

    # Considerando que "model" es el modelo ANFIS instanciado en cuestión
    # y que "train_loader" es el DataLoader con los datos de entrenamiento

    trainer(model, train_loader, verbose=True) # Esto ejecutará el entrenamiento del modelo

.. important::
    Recordar que el tamaño del batch de entrenamiento es definido por el DataLoader, por lo que es importante tener en cuenta este aspecto al momento de instanciarlo (ver :ref:`DataLoaders de PyTorch <DataLoaders_usage>`).