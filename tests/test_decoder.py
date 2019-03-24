from unittest import TestCase

import torch

from transformer.attention import MultiHeadAttention
from transformer.decoder import DecoderLayer, Decoder
from transformer.layers import PositionwiseFeedForward
from transformer.utils import subsequent_mask


class TestDecoderLayer(TestCase):
    def test_forward(self):
        # Parameters
        batch_size = 64
        sequence_length = 10
        d_k = d_v = d_model = input_size = 512
        d_ff = 2048
        N_decoder_layer = 6

        # initialization decoder
        decoder_layer = DecoderLayer(size=input_size,
                                     self_attn=MultiHeadAttention(n_head=8, d_model=d_model,
                                                                  d_k=d_k, d_v=d_v,
                                                                  dropout=0.1),
                                     memory_attn=MultiHeadAttention(n_head=8, d_model=d_model,
                                                                    d_k=d_k, d_v=d_v,
                                                                    dropout=0.1),
                                     feed_forward=PositionwiseFeedForward(d_model=d_model,
                                                                          d_ff=d_ff,
                                                                          dropout=0.1),
                                     dropout=0.1)

        decoder = Decoder(layer=decoder_layer, N=N_decoder_layer)

        # initialization input and memory
        x = torch.ones((batch_size, sequence_length, input_size))
        memory = torch.ones((batch_size, sequence_length, input_size))

        # subsequent mask: mask all words > i
        decoder_mask = subsequent_mask(sequence_length)

        # forward pass with fool input and memory (same here)
        out = decoder(x, memory, decoder_mask, None)

        # unit test
        self.assertEqual(out.shape, x.shape)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, memory.shape)
        self.assertEqual(x.shape, memory.shape)
