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

        transformer = Transformer(params)
