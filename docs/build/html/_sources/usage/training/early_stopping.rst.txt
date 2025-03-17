Early Stopping
==============

Todos los algoritmos de entrenamiento en Neuro-Fuzzy Toolbox tienen la opción de detenerse tempranamente si así se desea. Para ello, una instancia de la clase `EarlyStopping` 
debe ser pasada como argumento al momento de instanciar el algoritmo de entrenamiento.

.. note::
    - La clase `EarlyStopping` es una implementación de un mecanismo de parada temprana basado en la monitorización de una métrica de validación.
    - Para más detalles sobre su implementación en el toolbox, ver :ref:`Early Stopping`.

Instanciación
-------------
Para instanciar la clase `EarlyStopping`, se deben pasar los siguientes argumentos:

- `patience` (int): Número de épocas sin mejora en la métrica de validación después de las cuales el entrenamiento se detiene.
- `delta` (float): Umbral de mejora mínimo requerido en la métrica de validación para considerar que hay una mejora.

Ejemplo
#######
En el siguiente ejemplo, se muestra cómo instanciar la clase `EarlyStopping` con un `patience` de 10 y un `delta` de 0.001:

.. code:: python

    import neuro_fuzzy_toolbox as nft

    early_stopping = nft.EarlyStopping(patience=10, delta=0.001)