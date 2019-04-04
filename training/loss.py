import torch
import torch.nn as nn
from torch import Tensor


class LabelSmoothingLoss(nn.Module):
    """
    Wraps the :py:class:`torch.nn.KLDivLoss` loss with label smoothing.

    Reference: https://arxiv.org/abs/1512.00567

    .. note::

        "We propose a mechanism for encouraging the model to be less confident.
        While this may not be desired if the goal is to maximize the log-likelihood of training labels,
        it does regularize the model and makes it more adaptable.
        [...]
        Note that label smoothing achieves the desired goal of preventing the largest logit
        from becoming much larger than all others."



    Idea of label smoothing: Relax the confidence on the labels.

    Smooth the labels predicted probabilities towards 1 / n_classes.

    The equation (from tensorflow [1]_):

    .. math::

        new\_onehot\_labels = onehot\_labels \\cdot (1 - label\_smoothing) + \\frac{label\_smoothing}{num\_classes}


    .. [1] https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/losses/losses_impl.py#L706

    """

    def __init__(self, size, padding_token, smoothing=0.0):
        """
        Constructor of the `LabelSmoothingLoss` class.

        :param size: size of the output vocabulary set.

        :param padding_token: Padding token.

        :param smoothing: Smoothing factor.
        """
        # call base constructor
        super(LabelSmoothingLoss, self).__init__()

        # instantiate loss, ‘batchmean’: the sum of the output will be divided by batchsize
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_token = padding_token
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target) -> Tensor:
        """
        Forward pass of the LabelSmoothingLoss.

        :param x: predictions of the model, expected to contain log-probabilities.
        :param target: Ground truth, given as probabilities (i.e. without taking the logarithm).

        :return: loss
        """
        # ensure size of predictions of model matches given size at init
        assert x.size(1) == self.size, "The size of x ({}) doesn't match the given size in __init__ ({})".format(x.size(1), self.size)

        # apply label smoothing on the predictions
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        # padding mask
        mask = torch.nonzero(target.data == self.padding_idx)

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist

        return self.criterion(x, Tensor(true_dist, requires_grad=False))
