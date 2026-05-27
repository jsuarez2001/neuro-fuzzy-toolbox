Introduction
============

Neuro-Fuzzy Toolbox is a PyTorch-based library for the design, training, and
analysis of ANFIS-based neuro-fuzzy models within a modular and extensible
framework.

The toolbox implements three main model variants: the general
:class:`~neuro_fuzzy_toolbox.ANFIS`, the computationally efficient homogeneous
:class:`~neuro_fuzzy_toolbox.h_ANFIS`, and the :class:`~neuro_fuzzy_toolbox.rule_reduced_ANFIS`,
which avoids the combinatorial rule explosion associated with classical ANFIS
architectures. Training is supported through a hybrid algorithm, single-optimizer
training, and strategies with separate optimizers for premise and consequent
parameters. For rule-reduced models, the toolbox additionally provides a modified
SONFIS algorithm for structural adaptation through rule growing, splitting, and
pruning. Once a model is trained, built-in utilities allow inspection of the rule
base, visualization of membership functions, and estimation of local rule
contributions.

The toolbox supports two modes of use. In the first, users can directly employ
the provided models, training algorithms, and analysis utilities with minimal
setup. In the second, users familiar with PyTorch can reuse the implemented
layers and components to build custom architectures, define problem-specific
training procedures, or integrate neuro-fuzzy components into broader deep
learning pipelines.

The original ANFIS architecture is described in `Jang (1993) <https://doi.org/10.1109/21.256541>`_.

For installation instructions, see :ref:`Installation`.

For usage examples and workflows, see :ref:`Usage`.

For detailed API documentation, see :ref:`API Reference`.
