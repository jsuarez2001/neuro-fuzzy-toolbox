Basic Optimizer Training Algorithm
==================================
Algoritmo de entrenamiento basado en optimizadores. 

.. image:: ../../_static/Basic_optimizer_training_algorithm_pseudocode.png
    :width: 100%
    :align: center

Es aplicable a cualquier modelo personalizado que herede de la clase base nn.Module de PyTorch.

.. note::
    - Para más información sobre su implementación en el toolbox, ver :ref:`Basic Optimizer Based Training Algorithm`.
    - Puede ver ejemplos de uso :ref:`aquí <optimizer_training_examples>`.

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

    trainer = nft.Basic_optimizer_training_algorithm(
        epochs=epochs,
        loss_function=loss_fn,
        optimizer=optimizer,
        optimizer_params=params,
        validation=validation,
        early_stopping=early_stopping
    )

Ahora, asumiendo que ya se tiene un modelo instanciado y que se cuenta con un DataLoader de Pytorch con los datos de entrenamiento, se puede proceder con el entrenamiento del modelo:

.. code:: python

    # Considerando que "model" es el modelo instanciado en cuestión
    # y que "train_loader" es el DataLoader con los datos de entrenamiento
    trainer(model, train_loader)
