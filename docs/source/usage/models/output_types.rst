.. _output-types-usage:

Output Types on ANFIS Models
============================

.. important::

    All ANFIS model variants in Neuro-Fuzzy Toolbox
    (:ref:`ANFIS Variants <anfis-variants-usage>`) support everything
    presented in this section.

1. Multiple Outputs
-------------------
Neuro-Fuzzy Toolbox ANFIS models support multiple outputs, which is useful
for multivariate regression or multiclass classification problems.

.. note::

    When working with multiple outputs, each output has its own set of
    consequent parameters. This means the total number of consequent
    parameters in the model will be larger compared to a single-output model.
    The model structure adjusts automatically to handle multiple outputs.

    .. figure:: ../../_static/models/2_outputs_ANFIS.png
        :align: center
        :alt: homogeneous ANFIS structure

        2-output ANFIS model for 2-feature data, with 2 and 3 membership
        functions per feature respectively (source: authors).

To instantiate a multiple-output model, specify the number of outputs in the
*outputs* parameter:

.. code-block:: python

    # Simulating a dataset of 200 samples with 3 features
    x_train = 2 * torch.rand(200, 3) - 1  # shape must be (200, 3)

    model = nft.h_ANFIS(
        input_size=x_train.shape[1],  # 3 features
        num_mfs=2,                    # 2 MFs per feature
        outputs=2,                    # 2 outputs
        output_type='default'
    )

Consequents structure
^^^^^^^^^^^^^^^^^^^^^
As noted above, each output has its own set of consequent parameters:

.. code-block:: python

    dfs = model.get_consequents_structure()
    for i, df in enumerate(dfs):
        print(f"Output {i+1}:\n", df.to_string(), "\n")

.. code-block:: text

    Output 1:
                   x0        x1        x2          
                  c0        c1        c2        c3
    rule 1 -0.817876 -0.325063  0.288134  0.072033
    rule 2  0.640781 -0.439412  0.653194 -0.895495
    rule 3  0.174614 -0.089752  0.169705  0.287414
    rule 4  0.480423 -0.315141  0.127892  0.296729
    rule 5 -0.850491  0.862410 -0.573138  0.291747
    rule 6 -0.133427 -0.322040  0.804679 -0.113206
    rule 7 -0.307217 -0.596572  0.746054 -0.125493
    rule 8  0.845233 -0.224525  0.422965 -0.943465 

    Output 2:
                   x0        x1        x2          
                  c0        c1        c2        c3
    rule 1  0.040619  0.069412 -0.492848 -0.517831
    rule 2  0.816708 -0.994996  0.396029 -0.472854
    rule 3 -0.648234  0.285575 -0.392293 -0.075887
    rule 4  0.087970 -0.225077 -0.607382  0.759483
    rule 5 -0.962636 -0.884689 -0.591854  0.752209
    rule 6 -0.272644  0.370218  0.999882  0.213316
    rule 7 -0.120579 -0.277750  0.769877  0.588212
    rule 8  0.475320  0.466696 -0.418957 -0.983339

Outputs
^^^^^^^
The output tensor shape always follows this convention:

.. table::
    :align: center

    +-----------+-------------------------------------+
    | Outputs   | Output tensor shape                 |
    +===========+=====================================+
    | 1         | :math:`(batch\_size, )`             |
    +-----------+-------------------------------------+
    | 2 or more | :math:`(batch\_size, num\_outputs)` |
    +-----------+-------------------------------------+

For this example:

.. code-block:: python

    model(x_train[:5])

.. code-block:: text

    tensor([[ 0.0103, -0.2347],
            [-0.3309,  0.1947],
            [-0.6699, -0.4234],
            [-0.4590,  0.0759],
            [-0.6308, -0.5940]], grad_fn=<SqueezeBackward1>)

whereas for a single-output model:

.. code-block:: python

    single_output_model = nft.h_ANFIS(
        input_size=x_train.shape[1],  # 3 features
        num_mfs=2,                    # 2 MFs per feature
        outputs=1,                    # 1 output
        output_type='default'
    )

    single_output_model(x_train[:5])

.. code-block:: text

    tensor([ 0.0103, -0.3309, -0.6699, -0.4590, -0.6308],
           grad_fn=<SqueezeBackward1>)

.. _sigmoid-and-softmax-output:

2. Output Types
---------------
Neuro-Fuzzy Toolbox ANFIS models can be instantiated with different output
types via the *output_type* parameter:

- **'default'**: The model output is the default ANFIS output (the weighted
  sum of the outputs of each rule).
- **'sigmoid'**: A sigmoid layer is added at the model output.
- **'softmax'**: The model's forward method includes an optional softmax
  function, activated by the boolean attribute **return_probs**.

.. note::

    The reason a conditional softmax is used rather than a fixed softmax layer
    is related to how the *cross-entropy* loss function is implemented in
    PyTorch: it computes the softmax internally, so applying it explicitly in
    the model during training with this loss is unnecessary.

These options change the model structure and the behavior of the **predict**
method, as detailed below.

Default output
^^^^^^^^^^^^^^
With ``output_type='default'``, the model behaves as a regressor. The
**predict** method returns the raw numerical output of the **forward** method:

.. code-block:: python

    # Simulating a dataset of 200 samples with 2 features
    x_train = 2 * torch.rand(200, 2) - 1  # shape must be (200, 2)

Instantiating the model with 2 MFs per input feature:

.. code-block:: python

    model = nft.h_ANFIS(
        input_size=x_train.shape[1],  # 2 features
        num_mfs=2,                    # 2 MFs per feature
        outputs=1,                    # 1 output
        output_type='default'
    )

The *predict* method returns the raw model output (identical to *forward*):

.. code-block:: python

    model.predict(x_train[:5])

.. code-block:: text

    tensor([-0.0858,  0.2285,  0.4727,  0.6517,  0.8427])

.. code-block:: python

    model(x_train[:5])

.. code-block:: text

    tensor([-0.0858,  0.2285,  0.4727,  0.6517,  0.8427],
           grad_fn=<SqueezeBackward1>)

Sigmoid output
^^^^^^^^^^^^^^
With ``output_type='sigmoid'``, each model output behaves as a binary
classifier. A sigmoid layer is appended to the model, so the **forward**
output is a value in the range (0, 1).

.. note::

    This output type was introduced primarily for experimentation. In practice,
    the ``'softmax'`` output type is recommended for any classification problem.

Instantiating the model with 2 MFs per input feature:

.. code-block:: python

    model = nft.h_ANFIS(
        input_size=x_train.shape[1],  # 2 features
        num_mfs=2,                    # 2 MFs per feature
        outputs=1,                    # 1 output
        output_type='sigmoid'
    )

The **forward** output:

.. code-block:: python

    model(x_train[:5])

.. code-block:: text

    tensor([0.4289, 0.5286, 0.2916, 0.4935, 0.5071], grad_fn=<SigmoidBackward0>)

The *predict* method returns 1 if the positive-class probability exceeds 0.5,
and 0 otherwise:

.. code-block:: python

    model.predict(x_train[:5])

.. code-block:: python

    tensor([0, 1, 0, 0, 1])

This also applies with multiple outputs:

.. code-block:: python

    multiple_outputs_model = nft.h_ANFIS(
        input_size=x_train.shape[1],  # 2 features
        num_mfs=2,                    # 2 MFs per feature
        outputs=2,                    # 2 outputs
        output_type='sigmoid'
    )

.. code-block:: python

    multiple_outputs_model(x_train[:5])

.. code-block:: text

    tensor([[0.5557, 0.5229],
            [0.5073, 0.4143],
            [0.4351, 0.5481],
            [0.4641, 0.5223],
            [0.5815, 0.5450]], grad_fn=<SigmoidBackward0>)

.. code-block:: python

    multiple_outputs_model.predict(x_train[:5])

.. code-block:: text

    tensor([[1, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 1]])

.. _softmax-output-usage:

Softmax output
^^^^^^^^^^^^^^
With ``output_type='softmax'``, the model behaves as a multiclass classifier
(when more than 2 outputs are specified). The **forward** method accepts an
additional *return_probs* parameter: when set to ``True``, the outputs are
passed through a softmax function and class probabilities are returned;
otherwise, the raw unnormalized logits are returned. The number of classes
must be specified via the *outputs* parameter.

.. important::

    Using a single output with ``output_type='softmax'`` is not meaningful,
    as softmax is designed for multiclass classification. Two or more outputs
    are required.

Instantiating the model with 3 MFs per feature for a 4-class problem:

.. code-block:: python

    model = nft.h_ANFIS(
        input_size=x_train.shape[1],       # 2 features
        num_mfs=3,                         # 3 MFs per feature
        outputs=4,                         # 4 classes
        membership_function=nft.Gaussian_MF,
        output_type='softmax'
    )

The **forward** method returns the class logits by default:

.. code-block:: python

    model(x_train[:10])

.. code-block:: python

    tensor([[-0.0493, -0.1387, -0.0973, -0.2437],
            [ 0.0523,  0.0660, -0.3045,  0.0437],
            [-0.7400, -0.2403, -0.2109,  0.3169],
            [ 0.2379,  0.2125, -0.4648, -0.3233],
            [ 0.0642, -0.2672,  0.0778, -0.4195],
            [ 0.3088, -0.1348,  0.0422,  0.2339],
            [-0.0446, -0.0012, -0.3213, -0.0296],
            [ 0.0158, -0.2299,  0.0591, -0.3826],
            [-0.1415, -0.1499, -0.2126, -0.3147],
            [-0.1185, -0.2447,  0.3412, -0.1515]], grad_fn=<SqueezeBackward1>)

Passing *return_probs=True* returns the class probabilities after softmax:

.. code-block:: python

    model(x_train[:10], return_probs=True)

.. code-block:: python

    tensor([[0.2709, 0.2478, 0.2582, 0.2231],
            [0.2699, 0.2736, 0.1889, 0.2676],
            [0.1384, 0.2282, 0.2350, 0.3984],
            [0.3289, 0.3206, 0.1629, 0.1876],
            [0.2987, 0.2144, 0.3028, 0.1841],
            [0.2998, 0.1924, 0.2296, 0.2782],
            [0.2619, 0.2736, 0.1986, 0.2659],
            [0.2859, 0.2236, 0.2985, 0.1919],
            [0.2657, 0.2635, 0.2474, 0.2234],
            [0.2256, 0.1989, 0.3573, 0.2183]], grad_fn=<SoftmaxBackward0>)

The *predict* method returns the index of the class with the highest
probability:

.. code-block:: python

    model.predict(x_train[:10])

.. code-block:: python

    tensor([0, 1, 3, 0, 2, 0, 1, 2, 0, 2])