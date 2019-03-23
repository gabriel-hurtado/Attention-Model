import torch
import torch.nn as nn
from torch import Tensor

from utils import clones, BColors
from layers import PositionwiseFeedForward, LayerNormalization, ResidualConnection
from attention import ScaledDotProductAttention, MultiHeadAttention


class Decoder(nn.Module):
    """
    Implementation of the Decoder of the Transformer model.

    Constituted of a stack of N identical layers.
    """

    def __init__(self, layer, N):
        """
        Constructor for the global Decode
        :param layer: layer module to use.
        :param N: Number of layers to use.
        """
        # call base constructor
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNormalization(layer.size)

    def forward(self, x: Tensor, memory: Tensor, self_mask, memory_mask, verbose=False) -> Tensor:
        """
        Forward pass: Relays the output of layer i to layer i+1
        :param x: input Tensor of decoder
        :param memory: output from encoder
        :param self_mask: Mask to be used in the self-attention sub-module. Optional.
        :param memory_mask: Mask to be used in the memory-attention sub-module. Optional.
        :param verbose: Whether to add debug/info messages or not.
        :return:
        """
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'Going into layer {i}')
            x = layer(x, memory, self_mask, memory_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
        Implements one Decoder layer. The actual Decoder is a stack of N of these layers.

        The overall forward pass of this layer is as follows:

        ------------------------------> memory (Encoder output)
                                          |
                                          v
        x -> self-attn -> add & norm -> memory-attn -> add & norm -> feed_forward -> add & norm -> output
            |              ^           |               ^            |                ^
            v -----------> |           v ------------> |            v -------------> |
    """

    def __init__(self, size, self_attn, memory_attn, feed_forward, dropout):
        """
        Constructor for the ``DecoderLayer`` class.
        :param size: Input size
        :param self_attn: Class used for the self-attention part of the layer.
        :param memory_attn: Class used for the memory-attention part of the layer.
        :param feed_forward: Class used for the feed-forward part of the layer.
        :param dropout: dropout probability
        """

        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.memory_attn = memory_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(size, dropout), 3)

    def forward(self, x: Tensor, memory: Tensor, self_mask, memory_mask):
        """
        :param x: Input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).
                Should represent the input sentences or the output of the previous Encoder layer.
        :param memory: Memory input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).
                Should represent the memory output of the Encoder layer.
        :param self_mask: Mask to be used in the self-attention sub-module. Optional.
        :param memory_mask: Mask to be used in the memory-attention sub-module. Optional.
        :return: Output of the DecoderLayer, should be of the same shape as the input.
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))
        x = self.sublayer[1](x, lambda x: self.memory_attn(x, m, m, memory_mask))
        return self.sublayer[2](x, self.feed_forward)


if __name__ == '__main__':
    # parameters
    batch_size = 64
    sequence_length = 10
    d_k = d_v = 512

    # initialization decoder
    decoder_layer = DecoderLayer(size=512,
                                 self_attn=MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1),
                                 memory_attn=MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1),
                                 feed_forward=PositionwiseFeedForward(d_model=512, d_ff=2048, dropout=0.1),
                                 dropout=0.1)

    decoder = Decoder(layer=decoder_layer, N=6)

    # forward unit test
    x = torch.ones((64, 10, 512))
    memory = torch.ones((64, 10, 512))
    out = decoder(x, x, None, None)

    assert out.shape == x.shape
    assert out.shape == memory.shape
    assert x.shape == memory.shape
    print(out.shape)
    print(f'{BColors.OKGREEN}all unit tests passed')

