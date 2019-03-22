import copy
import torch.nn as nn


def clones(module, N):
    """
    Produce ``N`` identical copies of ``module`` and return them as a ``nn.ModuleList``.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
