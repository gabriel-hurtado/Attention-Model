import torch.nn as nn
from transformer.encoder import Encoder, EncoderLayer
from transformer.decoder import Decoder, DecoderLayer
from transformer.attention import MultiHeadAttention
from transformer.layers import PositionwiseFeedForward
from transformer.classifier import OutputClassifier
from transformer.embeddings import Embeddings, PositionalEncoding


class Transformer(nn.Module):
    """
    Main class for the Transformer model.
    Please see https://arxiv.org/abs/1706.03762 for the reference paper.
    """

    def __init__(self, params: dict):
        """
        Instantiate the ``Transformer`` class.

        :param params: Dict containing the set of parameters for the entire model\
         (e.g ``EncoderLayer``, ``DecoderLayer`` etc.) broken down in relevant sections, e.g.:

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

        """
        # call base constructor
        super(Transformer, self).__init__()

        # instantiate Encoder layer
        enc_layer = EncoderLayer(size=params['d_model'],
                                 self_attention=MultiHeadAttention(n_head=params['attention']['n_head'],
                                                                   d_model=params['d_model'],
                                                                   d_k=params['attention']['d_k'],
                                                                   d_v=params['attention']['d_v'],
                                                                   dropout=params['attention']['dropout']),
                                 feed_forward=PositionwiseFeedForward(d_model=params['d_model'],
                                                                      d_ff=params['feed-forward']['d_ff'],
                                                                      dropout=params['feed-forward']['dropout']),
                                 dropout=params['dropout'])

        # instantiate Encoder
        self.encoder = Encoder(layer=enc_layer, n_layers=params['N'])

        # instantiate Decoder layer
        decoder_layer = DecoderLayer(size=params['d_model'],
                                     self_attn=MultiHeadAttention(n_head=params['attention']['n_head'],
                                                                   d_model=params['d_model'],
                                                                   d_k=params['attention']['d_k'],
                                                                   d_v=params['attention']['d_v'],
                                                                   dropout=params['attention']['dropout']),
                                     memory_attn=MultiHeadAttention(n_head=params['attention']['n_head'],
                                                                   d_model=params['d_model'],
                                                                   d_k=params['attention']['d_k'],
                                                                   d_v=params['attention']['d_v'],
                                                                   dropout=params['attention']['dropout']),
                                     feed_forward=PositionwiseFeedForward(d_model=params['d_model'],
                                                                      d_ff=params['feed-forward']['d_ff'],
                                                                      dropout=params['feed-forward']['dropout']),
                                     dropout=params['dropout'])

        # instantiate Decoder
        self.decoder = Decoder(layer=decoder_layer, N=params['N'])

        pos_encoding = PositionalEncoding(d_model=params['d_model'], dropout=params['dropout'])

        self.src_embedings = nn.Sequential(Embeddings(d_model=params['d_model'], vocab_size=params['src_vocab_size']),
                                           pos_encoding)

        self.tgt_embedings = nn.Sequential(Embeddings(d_model=params['d_model'], vocab_size=params['tgt_vocab_size']),
                                           pos_encoding)

        self.classifier = OutputClassifier(d_model=params['d_model'], vocab=params['tgt_vocab_size'])

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self):
        pass
