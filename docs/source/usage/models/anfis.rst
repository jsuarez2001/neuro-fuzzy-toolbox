.. _ANFIS usage:

ANFIS
=====

Modelo ANFIS clásico. Con esta clase se permiten distintos números de funciones de membresía para cada feature de los datos de entrada.

Importación
-----------

.. code-block:: python

    import sys
    import os

    sys.path.append(os.path.abspath('../'))

    from neuro_fuzzy_toolbox import ANFIS
    import torch

Instanciación
-------------

Considerando que se tiene datos de entrada con 3 features:

.. code-block:: python

    x_train = 2 * torch.rand(200, 3) - 1

Antes de instanciar el modelo como tal, se debe definir la estructura de las funciones de membresía para cada feature de los datos de entrada. Para ello, se debe definir una lista, que contenga el número de funciones de membresía para cada feature.
Por ejemplo:

.. code-block:: python

    mf_distribution = [3, 2, 4]

En este caso, se está definiendo que el primer feature tiene 3 funciones de membresía, el segundo 2 y el tercero 4.

Luego, se puede instanciar el modelo utilizando la distribución definida.
Tener presente que el manejo de los tipos de salida (y su cantidad) es el mismo que en el modelo homogéneo (:ref:`h_ANFIS <h_ANFIS usage>`).

Por ejemplo:

- **Modelo ANFIS con 2 salidas para problemas de regresión**

.. code-block:: python

    model = ANFIS(
        mf_distribution=mf_distribution,
        outputs=2
    )

.. note::

    - El valor por defecto del parámetro `output_type` es `regression`.
    - El valor por defecto del parámetro `membership_function` es `GeneralizedBell_MF`.

- **Modelo ANFIS con 3 salidas para problemas de clasificación multiclase**

.. code-block:: python
    
    model = ANFIS(
        mf_distribution=mf_distribution,
        outputs=3,
        output_type='multiclass'
    )

- **Modelo ANFIS con 1 salida para problemas de clasificación binaria**

.. code-block:: python

    model = ANFIS(
        mf_distribution=mf_distribution,
        outputs=1
        output_type='binary'
    )

Todos los métodos y herramientas presentados con el modelo ANFIS homogéneo (:ref:`h_ANFIS <h_ANFIS usage>`) son válidos para el modelo ANFIS clásico, como las múltiples salidas y las aplicaciones a distintos tipos de problemas.

Pero existen ciertas consideraciones a tener en cuenta:

Parámetros de los antecedentes
------------------------------

Al contrario que en la clase h_ANFIS, en la clase ANFIS los parámetros de los antecedentes no se almacenan en un tensor único, debido a que el número de funciones de membresía puede variar para cada feature de los datos de entrada. 
Por lo tanto, los parámetros de los antecedentes se almacenan en una lista de parámetros de pytorch (torch.nn.ParameterList).

Esto implica que el método `get_premises()` no devolverá un tensor único, sino una lista de tensores, uno por cada feature de los datos de entrada.