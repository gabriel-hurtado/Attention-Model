import copy

import numpy as np
import torch
import torch.nn as nn


def clone(module, N) -> nn.ModuleList:
    """
    Produces ``N`` identical copies of ``module`` and returns them as a ``nn.ModuleList``.
    """
    return nn.ModuleList((copy.deepcopy(module),) * N)


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    :param size: Input size
    :return: Tensor with boolean mask on subsequent position
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class BColors:
    """
    Pre defined colors for console output
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
