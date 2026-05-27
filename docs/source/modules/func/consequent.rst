.. _Consequent Functions:

Consequent Functions
====================

This module defines the consequent functions used in the toolbox. These
functions define the output of each rule in neuro-fuzzy models.

All classes below inherit from the base class ``ConsequentFunction``,
which defines the common interface and serves as a reference for future
implementations.

.. note::
   Currently, only the standard linear consequent function of the ANFIS
   model is implemented. Adding new consequent function types would
   require non-trivial changes to the model and training algorithm
   implementations.

.. _Linear_CF:
.. autoclass:: neuro_fuzzy_toolbox.func.consequent.Linear_CF
   :members:
   :show-inheritance: