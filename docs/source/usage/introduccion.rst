Introduction
============

Una vez que se hayan seguido las indicaciones de :ref:`Installation`, se puede proceder al uso de las herramientas del toolbox.

Para importarlas, se puede hacer de la siguiente manera:

.. code-block:: python

    import sys
    import os

    sys.path.append(os.path.abspath('../'))

    import neuro_fuzzy_toolbox as nft

.. note::

    Dependiendo de la ubicación en la que se esté trabajando, la línea *sys.path.append(os.path.abspath('../'))* puede cambiar. En este caso, se asume que se está trabajando desde una carpeta contigua a la carpeta neuro_fuzzy_toolbox.
