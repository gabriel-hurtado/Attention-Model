import torch
from unittest import TestCase
from transformer.model import Transformer


class TestEncoder(TestCase):
    def test_forward(self):
        params = {
            'd_model': 512,
            'src_vocab_size': 27000,
            'tgt_vocab_size': 27000,

            'N': 6,
            'dropout': 0.1,

            'attention': {'n_head': 8,
                          'd_k': 64,
                          'd_v': 64,
                          'dropout': 0.1},

            'feed-forward': {'d_ff': 2048,
                             'dropout': 0.1},
        }

        # 1. test constructor
        transformer = Transformer(params)

        # 2. test forward pass
        batch_size = 64
        input_sequence_length = 10
        output_sequence_length = 13

        src_sequences = torch.ones((batch_size, input_sequence_length))
        tgt_sequences = torch.ones((batch_size, output_sequence_length))

        logits = transformer(src_sequences=src_sequences, src_masks=None, tgt_sequences=tgt_sequences, tgt_masks=None)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, torch.Size([batch_size, output_sequence_length, params['tgt_vocab_size']]))
        # check no nan values
        self.assertEqual(torch.isnan(logits).sum(), 0)
