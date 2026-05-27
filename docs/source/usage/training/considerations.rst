.. _Training considerations:

Initial Considerations
======================

Model inputs and outputs
------------------------
The models provided in Neuro-Fuzzy Toolbox are designed to follow this
input/output convention:

For inputs, data is expected to have the following dimensions:

.. table::
    :align: center

    +-----------+--------------------------------------+
    | Features  | Input data dimensions                |
    +===========+======================================+
    | 1         | :math:`(batch\_size, )`              |
    +-----------+--------------------------------------+
    | 2 or more | :math:`(batch\_size, num\_features)` |
    +-----------+--------------------------------------+

And for outputs:

.. table::
    :align: center

    +-----------+-------------------------------------+
    | Outputs   | Output tensor shape                 |
    +===========+=====================================+
    | 1         | :math:`(batch\_size, )`             |
    +-----------+-------------------------------------+
    | 2 or more | :math:`(batch\_size, num\_outputs)` |
    +-----------+-------------------------------------+

.. note::
    Multiple outputs refers not only to more than one output in the regression
    sense, but also to multiclass classification — ``num_outputs`` can be
    understood as the number of classes in a classification problem. See
    :ref:`Softmax Output <softmax-output-usage>` for details.

This convention should be kept in mind when defining DataLoaders for training
and evaluation.

.. _DataLoaders_usage:

PyTorch DataLoaders
-------------------
The training algorithms available in Neuro-Fuzzy Toolbox work with PyTorch
DataLoaders. These objects provide an efficient and straightforward way to
load training and validation data, and many of the internal mechanisms of the
training algorithms are designed to work with them and receive them as
arguments at runtime.

A simple example of how to define a DataLoader is shown below. For more
detailed information, refer to the official PyTorch documentation:
`Datasets & DataLoaders <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#>`_

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Simulating a training dataset with 200 samples and 3 features
    x_train = torch.rand(200, 3)
    y_train = torch.rand(200, )

When defining a DataLoader, the first argument must be a ``TensorDataset``
object wrapping the input and output data. The ``batch_size`` parameter
defines the number of samples used in each training iteration. Setting
``shuffle=True`` shuffles the data at each iteration; this is optional.

.. code-block:: python

    dataset = TensorDataset(x_train, y_train)

    # DataLoader with batch_size=32 and shuffle=True
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

Validation and test sets
-------------------------
Splitting data into training, validation, and test sets is the
**user's responsibility**. Neuro-Fuzzy Toolbox does not perform any automatic
data splitting.

To run the available training algorithms, a *train_loader* must be defined,
and a *val_loader* must be provided if a validation set is used. The user is
therefore responsible for splitting the data beforehand and defining the
corresponding datasets and DataLoaders:

.. code-block:: python

    train_dataset = TensorDataset(x_train, y_train)

    # DataLoader for training data with batch_size=32 and shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

.. code-block:: python

    val_dataset = TensorDataset(x_val, y_val)

    # DataLoader for validation data with batch_size=32
    val_loader = DataLoader(val_dataset, batch_size=32)

Then, assuming ``model`` and ``trainer`` are instances of an ANFIS model and
a training algorithm respectively, training is invoked as follows:

.. code-block:: python

    trainer(model, train_loader, val_loader)

Evaluation on the test set is left entirely to the user, who can use any
preferred library with the trained model.

Early Stopping
--------------
All training algorithms in Neuro-Fuzzy Toolbox support early stopping. To
enable it, an instance of the ``EarlyStopping`` class must be passed as an
argument when instantiating the training algorithm.

.. note::
    - ``EarlyStopping`` implements an early stopping mechanism based on
      monitoring a validation metric — specifically, the loss function used
      to initialize the training algorithm, evaluated on the
      **validation set**.
    - For more details on its implementation, see :ref:`Early Stopping`.

The ``EarlyStopping`` class takes the following arguments:

- **patience**: Number of epochs without improvement in the validation metric
  after which training is stopped.
- **delta**: Minimum improvement threshold required in the validation metric
  to be considered an improvement.

.. code-block:: python

    import neuro_fuzzy_toolbox as nft

    early_stopping = nft.EarlyStopping(patience=10, delta=0.001)

.. important::

    In Neuro-Fuzzy Toolbox, the ``EarlyStopping`` mechanism is **only**
    relevant when a validation set is defined and passed to the training
    algorithm. If no validation set is provided, early stopping has no effect
    on training.

Available training algorithms
------------------------------
The training algorithms available in Neuro-Fuzzy Toolbox are:

- :ref:`Hybrid Learning Algorithm <hybrid_learning_usage>`
- :ref:`Basic Optimizer Training Algorithm <basic_optimizer>`
- :ref:`Double Optimizer Training Algorithm <double_optimizer>`
- :ref:`SONFIS <sonfis-usage>`

For each algorithm, only two methods need to be considered:

- **Initialization** (*__init__*): Creates an instance of the algorithm,
  defining its parameters and any additional mechanisms.
- **Training** (*__call__*): Trains the model using the defined DataLoaders.

The following sections describe each training algorithm in detail, together
with usage examples.