.. _Examples:

Examples
========

This section provides four end-to-end examples covering the main workflows
supported by Neuro-Fuzzy Toolbox. Each example includes data loading and
preprocessing, model instantiation, parameter initialization, training, and
evaluation. The examples are ordered roughly by complexity, from a standard
classifier built with a built-in training algorithm to structural adaptation
with SONFIS on regression data.

The datasets used are publicly available through the
`UCI Machine Learning Repository <https://archive.ics.uci.edu/>`_ or
generated synthetically. Data preprocessing, train/validation/test splitting,
and DataLoader construction follow standard scikit-learn and PyTorch
conventions.

- **Example 1** covers multiclass classification on the Iris dataset using
  an ``h_ANFIS`` model trained with the
  :ref:`Basic Optimizer Training Algorithm <basic_optimizer>`, and
  demonstrates post-training rule inspection with
  :ref:`RulesAnalyzer <Rule_Analyzer-class>`.

- **Example 2** covers multiclass classification on the Glass Identification
  dataset using a ``rule_reduced_ANFIS`` model, combining an initial
  gradient-based training phase with a custom greedy rule-growing procedure
  built on the :ref:`low-level API <custom-training>`.

- **Example 3** covers binary classification on the Heart Disease dataset
  using a ``rule_reduced_ANFIS`` model trained with
  :ref:`SONFIS <sonfis-usage>` for structural adaptation.

- **Example 4** covers regression of a noisy 3D surface using a
  ``rule_reduced_ANFIS`` model trained with
  :ref:`SONFIS <sonfis-usage>`, including surface visualizations of the
  target function and model predictions.

.. toctree::
   :maxdepth: 2

   examples/example1
   examples/example2
   examples/example3
   examples/example4