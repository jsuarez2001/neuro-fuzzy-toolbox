.. _SONFIS:

SONFIS
========

Este módulo contiene 2 implementaciones de la red SONFIS (Self-Organizing Neuro-Fuzzy Inference System) que es una red auto-organizada que combina el algoritmo para la actualización de parámetros clásico del modelo ANFIS con
una serie de operaciones para el crecimiento y poda de subredes. Este método solo es aplicable a modelos ANFIS homogéneos con reglas reducidas.

Para más detalles sobre el algoritmo SONFIS, puede consultar el artículo original: `SONFIS: Structure Identification and Modeling with a Self-Organizing Neuro-Fuzzy Inference System <https://doi.org/10.1080/18756891.2016.1175809>`_.

.. image:: ../../_static/SONFIS_pseudocode.png
   :alt: Pseudocódigo del algoritmo SONFIS.
   :align: center
   :width: 600px

.. raw:: html

   <br>

.. raw:: latex

   \newline

SONFIS
------

.. autoclass:: neuro_fuzzy_toolbox.training.sonfis.SONFIS
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __call__

alternative SONFIS
------------------

.. autoclass:: neuro_fuzzy_toolbox.training.sonfis.alt_SONFIS
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __call__