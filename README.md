# Neuro-Fuzzy Toolbox

A PyTorch-based library for the design, training, and analysis of ANFIS-based
neuro-fuzzy models. The toolbox provides ready-to-use model variants and training
algorithms, a structural adaptation algorithm, and utilities for rule inspection
and local contribution analysis. Its modular design also makes it a flexible
basis for building custom training procedures and deep neuro-fuzzy architectures.

## Features

- **Three ANFIS model variants**: classical `ANFIS`, homogeneous `h_ANFIS`, and
  `rule_reduced_ANFIS` for high-dimensional settings.
- **Multiple training strategies**: hybrid learning algorithm, single-optimizer
  training, and dual-optimizer training with independent premise and consequent
  optimizers. All strategies integrate with PyTorch loss functions and support
  early stopping.
- **Structural adaptation**: a modified SONFIS algorithm for `rule_reduced_ANFIS`
  models, supporting rule growing, splitting, and pruning during training.
- **Rule inspection and analysis**: tabular export of premises and consequents,
  membership function visualization, intermediate layer access, and local
  rule-contribution analysis via `RulesAnalyzer`.
- **Low-level API**: direct access to premise and consequent parameter subsets
  for custom optimizer instantiation, and programmatic rule addition and removal
  at runtime.

## Requirements
 - torch >= 2.5 (tested in 2.5.1)
 - numpy >= 2.2 (tested in 2.2.1)
 - pandas >= 2.2 (tested in 2.2.3)
 - matplotlib >= 3.10 (tested in 3.10.0)


## Documentation

Full documentation including a usage guide, API reference, and end-to-end
examples is available at: *(link pending)*