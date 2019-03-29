import torch
import torch.nn as nn
from transformer.utils import subsequent_mask
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

    def forward(self, src_sequences, src_masks, tgt_sequences, tgt_masks):
        """
        DISCLAIMER: There are missing parts / bugs in this forward for certain.
        Have to identify & fix them.

        :param src_sequences: Batch of input sentences. Should be of shape (batch_size, in_seq_len).

        :param  src_masks: Mask, hiding the padding in the input batch. Should be same shape as src_sequences.

        :param tgt_sequences: Batch of output sentences. Should be of shape (batch_size, out_seq_len).

        :param tgt_masks: Mask, hiding the padding in the output batch. TODO: Shape>

        :return: Logits, of shape (batch_size, out_seq_len, d_model)
        """

        # 1. embed the input batch
        src_sequences = self.src_embedings(src_sequences.type(torch.LongTensor))

        # 2. encoder stack
        encoder_output = self.encoder(src_sequences.type(torch.FloatTensor))

        # 3. get subsequent mask to hide subsequent positions in the decoder.
        self_mask = subsequent_mask(tgt_sequences.shape[1])

        # 4. embed the output batch
        tgt_sequences = self.tgt_embedings(tgt_sequences.type(torch.LongTensor))

        # 4. decoder stack
        decoder_output = self.decoder(x=tgt_sequences.type(torch.FloatTensor), memory=encoder_output, self_mask=self_mask, memory_mask=None)

        # 5. classifier
        logits = self.classifier(decoder_output)

        return logits
