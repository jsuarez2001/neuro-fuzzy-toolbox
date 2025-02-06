.. _h_ANFIS usage:

ANFIS homogéneo
================

Modelo ANFIS cuyo número de funciones de membresía es igual para todos los features de los datos de entrada.

Importación
-----------

Se importará la clase **h_ANFIS** y la función de membresía **Gaussian_MF** para los ejemplos a continuación.

.. code-block:: python

    import sys
    import os

    sys.path.append(os.path.abspath('../'))

    from neuro_fuzzy_toolbox import h_ANFIS, Gaussian_MF
    import torch

.. code-block:: python

    # Se simulará un conjunto de datos de entrada con 3 features
    x_train = 2 * torch.rand(200, 3) - 1


Instanciación
-------------
A continuación se instancia un modelo h_ANFIS con 3 funciones de membresía para cada feature de los datos de entrada y con una salida. Se mostrarán todos los parámetros relevantes para instanciar la clase.

.. code-block:: python

    model = h_ANFIS(
        input_size=x_train.shape[1], # 3 features
        num_mfs=3, # 3 funciones de membresía
        outputs=1, # 1 salida
        membership_function=Gaussian_MF, # Función de membresía gaussiana
        output_type='regression', # Tipo de salida: regresión
    )


.. note::

    Por defecto, la función de membresía es GeneralizedBell_MF, pero se trabajará con Gaussian_MF en este ejemplo. para más detalles sobre las funciones de membresía, ver :ref:`Membership Functions <Membership Functions>`.

El método *plot_premises()* permite visualizar las funciones de membresía de los antecedentes del modelo.

.. code-block:: python

    model.plot_premises()


.. image:: ../../_static/h_anfis_plot_premises.png
    :align: center

.. code-block:: python

    model.plot_premises(group_by_dim=True)

.. image:: ../../_static/h_anfis_plot_premises_grouped.png
    :align: center

Inicialización de parámetros
----------------------------
Ahora mismo el modelo se encuentra con todos sus parámetros inicializados aleatoriamente.
En caso de que se estime conveniente, es posible inicializar los parámetros de los antecedentes en base a los datos con los que se trabajará:

.. code-block:: python

    model.init_premises(x_train)

    model.plot_premises(group_by_dim=True)


.. image:: ../../_static/h_anfis_plot_init_premises_grouped.png
    :align: center

Para visualizar los parámetros en cuestión se puede utilizar el método **show_premises_structure()**, el cual imprime por pantalla un dataframe con los valores de los parámetros de las funciones de membresía:

.. code-block:: python

    model.show_premises_structure()

.. code-block:: python

           mu (x0)  sigma (x0)   mu (x1)  sigma (x1)   mu (x2)  sigma (x2)
    MF 0 -0.990143    0.497452 -0.986824    0.494897 -0.983210    0.494364
    MF 1  0.004761    0.497452  0.002970    0.494897  0.005518    0.494364
    MF 2  0.999665    0.497452  0.992764    0.494897  0.994247    0.494364

El método **model.premises_structure** retorna el dataframe correspondiente, mientras que el método **model.get_premises()** retorna un tensor con los valores correspondientes.

En cuanto a los consecuentes, hay métodos análogos a los de los antecedentes: **show_consequents_structure()**, **model.consequents_structure** y **get_consequents()**. 

La única diferencia es que, en el caso del método **model.consequents_structure**, se retorna una lista con los dataframe correspondientes a cada una de las salidas del modelo. En este caso, como solo hay una salida, se retorna una lista con un solo dataframe.

.. code-block:: python

    model.show_consequents_structure()

.. code-block:: python

    - Output 1:
              c0 (x0)   c1 (x1)   c2 (x2)        c3
    rule 1  -0.402982  0.913287 -0.147453 -0.570277
    rule 2  -0.558947 -0.600798 -0.186016 -0.840755
    rule 3   0.168397 -0.883966  0.474520  0.312040
    rule 4  -0.122751  0.630311 -0.797790  0.563108
    rule 5   0.361740 -0.286652 -0.325836  0.847129
    rule 6  -0.456639 -0.769045 -0.617894  0.075701
    rule 7  -0.118061 -0.842846 -0.196719 -0.067862
    rule 8  -0.775172  0.827982 -0.956070  0.850902
    rule 9  -0.781440 -0.009967  0.381071  0.127973
    rule 10  0.578195  0.767374  0.944861  0.173381
    rule 11 -0.579265  0.665136 -0.931385  0.141964
    rule 12 -0.677500 -0.522054 -0.871355 -0.329690
    rule 13 -0.931001 -0.564158  0.425146 -0.055656
    rule 14 -0.163182  0.894706  0.172442  0.128253
    rule 15 -0.343025  0.517860  0.329148  0.808215
    rule 16  0.975743  0.005502 -0.814048  0.887746
    rule 17 -0.015661 -0.065317 -0.171591  0.917433
    rule 18 -0.682314  0.237104  0.514205  0.443589
    rule 19 -0.061958 -0.398458  0.066680  0.805317
    rule 20 -0.109509  0.672676  0.771990  0.259475
    rule 21 -0.887838  0.148851 -0.494752  0.740001
    rule 22 -0.506924  0.124466 -0.304620  0.406688
    rule 23  0.096541  0.600472  0.615980 -0.438880
    rule 24  0.421969  0.594888 -0.104359  0.558902
    rule 25  0.454787  0.143593 -0.386165 -0.442262
    rule 26 -0.872414  0.364831  0.624177  0.791676
    rule 27  0.880137 -0.213823 -0.404114 -0.578382

Método predict vs forward
--------------------------

El método **forward** permite obtener la salida del modelo para un conjunto de datos de entrada en forma de un tensor de PyTorch. Este método genera el grafo de cómputo de PyTorch, lo que permite realizar backpropagation y entrenar el modelo.

.. code-block:: python

    model(x_train[:10])

.. code-block:: python

    tensor([ 0.3339, -0.0306, -0.1787, -0.3885,  0.7853, -0.6146,  0.0076, -0.4021, -0.1976,  0.3151], grad_fn=<SqueezeBackward1>)

Para evitar generar el grafo de cómputo de PyTorch, se puede utilizar el método **no_grad** de PyTorch.

.. code-block:: python

    with torch.no_grad():
        output = model(x_train[:10])

    print(output)

.. code-block:: python
    
    tensor([ 0.3339, -0.0306, -0.1787, -0.3885,  0.7853, -0.6146,  0.0076, -0.4021, -0.1976,  0.3151])

El método **predict** retorna la predicción del modelo para un conjunto de datos de entrada en forma de un array de numpy sin generar el grafo de cómputo de PyTorch (como lo hace el método forward). Dependiendo del tipo de salida del modelo se realizan ciertas operaciones para entregar el resultado.

.. code-block:: python

    pred = model.predict(x)

Si el modelo es de tipo:

- Regresión: Se retorna el valor de la salida directamente.
- Clasificación binaria: Se retorna 1 si la probabilidad de la clase positiva es mayor a 0.5, 0 en caso contrario. (La salida pasa por una función sigmoide).
- Clasificación multiclase: Se retorna el índice de la clase con mayor probabilidad. (La salida pasa por una función softmax y posteriormente se elige la clase con mayor probabilidad).

Múltiples salidas
-----------------
La única diferencia en la arquitectura del modelo al trabajar con múltiples salidas es la cantidad de parámetros consecuentes generados. 

.. code-block:: python

    x_train = 2 * torch.rand(200, 2) - 1 # ejemplo de datos de entrenamiento con 2 features

    model = h_ANFIS(
        input_size=x_train.shape[1], # 3 features
        num_mfs=3, # 3 funciones de membresía
        outputs=2, # 2 salidas
        membership_function=Gaussian_MF, # Función de membresía gaussiana
        output_type='regression', # Tipo de salida: regresión
    )

.. code-block:: python

    model.show_consequents_structure()

.. code-block:: python

    - Output 1:
             c0 (x0)   c1 (x1)        c2
    rule 1  0.412696  0.522680  0.090145
    rule 2 -0.976030 -0.419625  0.283456
    rule 3  0.254746  0.443201 -0.872905
    rule 4  0.790756  0.528562  0.858889
    rule 5  0.651066 -0.942244 -0.114714
    rule 6 -0.897736  0.491714 -0.059783
    rule 7 -0.075024  0.502939 -0.868872
    rule 8 -0.819807 -0.380666  0.491858
    rule 9  0.367133 -0.824872  0.538071
    
    
    - Output 2:
             c0 (x0)   c1 (x1)        c2
    rule 1 -0.682649  0.267264  0.636341
    rule 2  0.402898 -0.281733 -0.870279
    rule 3  0.929400  0.168972  0.334629
    rule 4 -0.860564  0.804722  0.599917
    rule 5  0.660574  0.454537  0.582051
    rule 6  0.882556  0.788579  0.761563
    rule 7 -0.752488 -0.172638  0.551565
    rule 8 -0.063006 -0.112478 -0.040473
    rule 9  0.690270  0.598800 -0.420998

Ahora la salida del modelo se vería de la siguiente manera:

.. code-block:: python

    model(x_train[:10])

.. code-block:: python

    tensor([[-0.2550,  0.4605],
            [ 0.2885,  0.1500],
            [-0.3057,  0.2601],
            [ 0.5863,  0.2624],
            [ 0.1534,  0.4544],
            [-0.4603,  0.4648],
            [ 0.4690,  0.4441],
            [-0.1083,  0.2348],
            [ 0.0827,  0.1670],
            [-0.2726,  0.4799]], grad_fn=<SqueezeBackward1>)

Problemas multiclase
--------------------

Para trabajar con clasificación multiclase, se debe instanciar el modelo con el parámetro *output_type='multiclass'* y el número de clases correspondientes al problema se deben especificar en el parámetro *outputs*.

Suponiendo que se cuenta con un problema de 3 clases, la instanciación del modelo sería la siguiente:

.. code-block:: python

    x_train = 2 * torch.rand(200, 2) - 1 # ejemplo de datos de entrenamiento con 2 features

    model = h_ANFIS(
        input_size=x_train.shape[1], # 3 features
        num_mfs=2, # 2 funciones de membresía
        outputs=3, # 3 salidas
        membership_function=Gaussian_MF, # Función de membresía gaussiana
        output_type='multiclass', # Tipo de salida: clasificación multiclase
    )

Una comparación de las posibles salidas del método forward y el método predict se puede ver a continuación: 

- **forward**: Retorna los logits sin normalizar de las clases.

.. code-block:: python

    model(x_train[:10])

.. code-block:: python

    tensor([[-0.5759,  0.9541,  0.2970],
            [-0.2917,  0.4547, -0.9439],
            [ 0.5124, -0.6826, -0.3454],
            [-0.8390,  0.9174, -0.9471],
            [-0.9870,  1.0792, -0.8661],
            [-1.7170,  1.9342, -0.3654],
            [-0.0322,  0.5905,  0.7484],
            [ 0.1828, -0.0688, -1.1221],
            [-0.4592,  0.7823, -0.2416],
            [-0.3206,  0.5607, -0.6278]], grad_fn=<SqueezeBackward1>)

- **forward con return_probabilities=True**: Retorna las probabilidades de las clases (los logits son pasados por una función softmax).

.. code-block:: python

    model(x_train[:10], return_probabilities=True)

.. code-block:: python

    tensor([[0.1248, 0.5764, 0.2988],
            [0.2755, 0.5811, 0.1435],
            [0.5791, 0.1753, 0.2456],
            [0.1301, 0.7532, 0.1167],
            [0.0998, 0.7876, 0.1126],
            [0.0231, 0.8879, 0.0891],
            [0.1981, 0.3693, 0.4325],
            [0.4881, 0.3795, 0.1324],
            [0.1753, 0.6067, 0.2179],
            [0.2410, 0.5817, 0.1772]], grad_fn=<SoftmaxBackward0>)

- **predict**: Retorna la predicción del modelo en forma de array de numpy (los índices correspondientes a la clase con mayor probabilidad)

.. code-block:: python

    model.predict(x_train[:10])

.. code-block:: python

    array([1, 1, 0, 1, 1, 1, 2, 0, 1, 1])
