import math

from torch import nn


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Creates a word embeddings.

        Note that for the transformer model, the input embeddings
        is contrained to using the same weight matrix as the output transformation.

        :param d_model: The dimension of the output to use.
        :param vocab_size: The size of the vocabulary.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model_sqrt = math.sqrt(d_model)

    def forward(self, x):
        return self.embeddings(x) * self.d_model_sqrt
