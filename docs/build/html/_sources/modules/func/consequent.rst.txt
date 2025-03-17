.. _Consequent Functions:

Consequent Functions
====================

Aquí están definidas las funciones de consecuente utilizadas en el Toolbox. Estas funciones son utilizadas para definir las salidas de los modelos neurodifusos.

Las clases aquí presentes heredan de la clase base llamada *ConsequentFunction*, la cual solo define su estructura y sirve de guía para futuras implementaciones.

.. note::
   De momento solo se encuentra implementada la función consecuente lineal típica del modelo ANFIS. La implementación de nuevas funciones consecuentes implicaría modificaciones profundas en la estructura de los modelos y algoritmos implementados.

.. _Linear_CF:
.. autoclass:: neuro_fuzzy_toolbox.func.consequent.Linear_CF
   :members:
   :show-inheritance: