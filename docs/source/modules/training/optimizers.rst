.. _Optimizers:

Optimizers
==========

Optimizers implemented in Neuro-Fuzzy Toolbox. They follow the
``torch.optim.Optimizer`` interface, so they can be passed to any training
algorithm of the toolbox through its ``optimizer`` argument.

.. _NormalizedGradientDescent:

.. autoclass:: neuro_fuzzy_toolbox.training.optimizers.NormalizedGradientDescent
   :members: step
   :show-inheritance:
