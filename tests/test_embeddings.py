from unittest import TestCase

import numpy as np
import torch

from transformer.embeddings import Embeddings


class TestEmbeddings(TestCase):
    def test_forward(self):
        out_size = np.random.randint(10, 100)
        vocab_size = np.random.randint(100, 1000)
        nb_words = np.random.randint(10, 100)

        # Generate input and one-hot-encode it
        input = torch.randint(0, vocab_size, (nb_words,))
        one_hots = torch.eye(vocab_size, dtype=torch.long)[input, :]

        # Create model
        embeddings = Embeddings(d_model=out_size, vocab_size=vocab_size)

        # Run forward pass
        out = embeddings.forward(one_hots)

        # Check output
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (nb_words, vocab_size, out_size))
