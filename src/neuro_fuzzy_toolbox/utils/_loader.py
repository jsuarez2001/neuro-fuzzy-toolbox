import torch
from torch.utils.data import DataLoader, TensorDataset


def get_loader_tensors(loader):
    """
    Returns the complete input and target tensors behind a DataLoader.

    The training algorithms and the SONFIS structural operators need the whole
    dataset at once (for instance, to solve the least-squares problem of the
    consequents or to assign every sample to its most active rule).

    If the DataLoader wraps a :class:`torch.utils.data.TensorDataset`, its tensors are
    returned directly, at no cost. Any other dataset (a custom ``Dataset``, a ``Subset`` obtained from
    :func:`torch.utils.data.random_split`, etc.) is read in a single ordered pass, using the
    ``collate_fn`` of the original DataLoader. The order of the samples returned
    is always the order of the dataset, regardless of the ``shuffle`` and ``drop_last`` options of the loader.

    Note:
        For datasets that are not a ``TensorDataset``, the whole dataset is loaded in memory on each call.

    Args:
        loader (DataLoader): DataLoader whose dataset yields ``(inputs, targets)`` pairs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Inputs of shape ``(n_samples, input_size)`` and targets.
    """
    dataset = loader.dataset
    if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
        return dataset.tensors[0], dataset.tensors[1]

    full_pass = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                           collate_fn=loader.collate_fn, num_workers=0)
    x, y = next(iter(full_pass))[:2]
    return x, y
