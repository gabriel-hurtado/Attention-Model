import copy
import torch.nn as nn


def clones(module, N):
    """
    Produce ``N`` identical copies of ``module`` and return them as a ``nn.ModuleList``.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
