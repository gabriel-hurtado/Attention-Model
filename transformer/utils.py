import copy

import numpy as np
import torch
import torch.nn as nn

# if CUDA available, moves computations to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def clone(module, N) -> nn.ModuleList:
    """
    Produces ``N`` identical copies of ``module`` and returns them as a ``nn.ModuleList``.
    """
    return nn.ModuleList((copy.deepcopy(module),) * N)


def subsequent_mask(size):
    """
    Masks out subsequent positions.
    :param size: Input size
    :return: Tensor with boolean mask on subsequent position
    """
    attn_shape = (1, size, size)
    # pylint: disable=no-member
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask).float().to(device) == 0
