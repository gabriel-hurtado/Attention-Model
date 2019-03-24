import torch.nn as nn
from torch import Tensor

from layers import ResidualConnection
from utils import clones


class Encoder(nn.Module):
    """
    Implementation of the Encoder of the Transformer model.

    Constituted of a stack of N identical layers.
    """

    def __init__(self, layer: nn.Module, n_layers: int):
        """
        Constructor for the global Encoder.

        :param layer: layer type to use.
        :param n_layers: Number of layers to use.
        """
        # call base constructor
        super().__init__()
        self.layers = clones(layer, n_layers)

    def forward(self, x: Tensor, mask=None, verbose=False) -> Tensor:
        """
        Implements the forward pass: relays the output of layer i to layer i+1.

        :param x: Input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).
                Should represent the input sentences.
        :param mask: Mask to use in the layers. Optional.
        :param verbose: Whether to add debug/info messages or not.
        :return: Output tensor, should be of same shape as input.
        """
        for i, layer in enumerate(self.layers):
            if verbose:
                print('Going into layer {}'.format(i + 1))
            x = layer(x, mask)

        return x


class EncoderLayer(nn.Module):
    """
    Implements one Encoder layer. The actual Encoder is a stack of N of these layers.

    The overall forward pass of this layer is as follows:

    input x -> Self-Attention -> Sum -> LayerNorm -> FeedForward -> Sum -> LayerNorm -> output
            |                            ^                |                  ^
            v -------------------------> |                v ---------------- |
    """

    def __init__(self, size: int, self_attention: nn.Module, feed_forward: nn.Module,
                 dropout: float):
        """
        Constructor for the ``EncoderLayer`` class.

        :param size: Input size.
        :param self_attention: Class used for the self-attention part of the layer.
        :param feed_forward: Class used for the feed-forward part of the layer.
        :param dropout: Dropout probability.
        """
        # call base constructor
        super().__init__()

        # get self-attn & feed-forward sub-modules
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.size = size

        self.sublayer = clones(ResidualConnection(size, dropout), 2)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        Implements the forward pass of the Encoder layer.

        :param x: Input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).
                Should represent the input sentences or the output of the previous Encoder layer.

        :param mask: Mask to be used in the self-attention sub-module. Optional.

        :return: Output of the EncoderLayer, should be of the same shape as the input.

        """
        attention_out = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))

        return self.sublayer[1](attention_out, self.feed_forward)
