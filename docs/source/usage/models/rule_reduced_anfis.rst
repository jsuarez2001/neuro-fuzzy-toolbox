.. _rule-reduced ANFIS usage:

rule-reduced ANFIS
==================

Modelo ANFIS con reglas reducidas. Es una variante del modelo ANFIS homogéneo (:ref:`mas información de los modelos implementados. <ANFIS>`) que reduce el número de reglas y, por ende, el número de parámetros concecuentes en el modelo ANFIS.

A continuación se muestra una comparación entre el modelo ANFIS homogéneo y el modelo ANFIS con reglas reducidas, ambos con 2 reglas y 3 funciones de pertenencia por variable de entrada:

- **Modelo ANFIS homogéneo**

.. figure:: ../../_static/h_anfis.png
   :align: center
   :width: 600px
   :alt: h_ANFIS example

- **Modelo rule-reduced ANFIS**

.. figure:: ../../_static/rule_reduced_anfis.png
   :align: center
   :width: 600px
   :alt: Rule-Reduced ANFIS example


Instanciación
-------------

Para su instanciación, la clase h_ANFIS tiene un parámetro llamado `rule_reduced`, el cual debe ser establecido en `True` (Por defecto es `False`):

.. code-block:: python

    from neuro_fuzzy_toolbox import h_ANFIS

    x_train = 2 * torch.rand(200, 2) - 1 # ejemplo de datos de entrenamiento con 2 features

    model = h_ANFIS(
        input_size=2,
        num_mfs=3,
        outputs=1
    )

Todos los métodos y herramientas presentados con el modelo ANFIS homogéneo (:ref:`h_ANFIS <h_ANFIS usage>`) son válidos para el modelo ANFIS con reglas reducidas, como las múltiples salidas y las aplicaciones a distintos tipos de problemas.
