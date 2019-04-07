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

    Smooth the labels predicted probabilities towards 1 / n_classes. This works by modifying the ground truth probabilities.

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
        assert 0.0 < smoothing <= 1.0, "The smoothing factor should be in [0, 1], got {}.".format(smoothing)

        # call base constructor
        super(LabelSmoothingLoss, self).__init__()

        self.size = size

        # instantiate loss, ‘batchmean’: the sum of the output will be divided by batchsize
        self.criterion = nn.KLDivLoss(reduction='batchmean')

        # LogSoftmax layer
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # padding token to ignore
        self.padding_token = padding_token

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing / (size - 2)  # exclude pad and true label

        # create & save a tensor of size (1, size) containing the smoothing value everywhere
        # will be used as a basis to create label-smoothed targets:
        # - simply have to add confidence at true index and ignore padding
        smoothed_targets = torch.full((size,), self.smoothing)
        smoothed_targets[self.padding_token] = 0
        self.register_buffer('smoothed_targets', smoothed_targets.unsqueeze(0))  # (1, size)

    def forward(self, x, targets) -> Tensor:
        """
        Forward pass of the LabelSmoothingLoss.

        :param x: predictions of the model (i.e. raw class scores), of shape [batch_size, seq_length, vocabulary_size]
        :param targets: Ground truth tokens indices, of shape [batch_size, seq_length]

        :return: loss value
        """
        # ensure size of predictions of model matches given size at init
        assert x.shape[1] == self.size, "The size of x ({}) doesn't match the given size in __init__ ({})".format(x.shape[1], self.size)

        batch_size, seq_len, vocabulary_size = x.size()

        # go through LogSoftmax layer and flatten out the tensors for simplicity
        outputs_log_softmax = self.log_softmax(x)
        outputs_flat = outputs_log_softmax.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        # repeat the smoothed_targets tensor as necessary to match batch size
        smoothed_targets = self.smoothed_targets.repeat(targets_flat.size(0), 1)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        # Copies self.confidence into smoothed_targets at indices specified by targets_flats
        # dim (int) – the axis along which to index
        smoothed_targets.scatter_(dim=1, index=targets_flat.unsqueeze(1), value=self.confidence)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        # Fills elements of smoothed_targets with 0s where targets_flats is equal to pad index
        # The shape of mask must be broadcastable with the shape of the underlying tensor.
        smoothed_targets.masked_fill_(mask=(targets_flat == self.padding_token).unsqueeze(1), value=0)
        # masked_targets: (batch_size * seq_len, vocabulary_size)

        # Finally, go through loss function
        loss = self.criterion(outputs_flat, smoothed_targets)

        return loss
