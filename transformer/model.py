import torch
import torch.nn as nn
from datetime import datetime
from transformer.utils import subsequent_mask
from transformer.encoder import Encoder, EncoderLayer
from transformer.decoder import Decoder, DecoderLayer
from transformer.attention import MultiHeadAttention
from transformer.layers import PositionwiseFeedForward
from transformer.classifier import OutputClassifier
from transformer.embeddings import Embeddings, PositionalEncoding

# if CUDA available, change some tensor types to move computations to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    device = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


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

    # pylint: disable=unused-argument
    def forward(self, src_sequences, src_masks, tgt_sequences, tgt_masks) -> torch.Tensor:
        """
        DISCLAIMER: There are missing parts / bugs in this forward for certain.
        Have to identify & fix them.

        :param src_sequences: Batch of input sentences. Should be of shape (batch_size, in_seq_len).

        :param  src_masks: Mask, hiding the padding in the input batch. Should be same shape as src_sequences.

        :param tgt_sequences: Batch of output sentences. Should be of shape (batch_size, out_seq_len).

        :param tgt_masks: Mask, hiding the padding in the output batch. TODO: Shape>

        :return: Logits, of shape (batch_size, out_seq_len, d_model)
        """

        # 1. embed the input batch: have to move input sequences to torch.*.LongTensor
        src_sequences = self.src_embedings(src_sequences.type(LongTensor))

        # 2. encoder stack
        encoder_output = self.encoder(src_sequences)

        # 3. get subsequent mask to hide subsequent positions in the decoder.
        self_mask = subsequent_mask(tgt_sequences.shape[1])

        # 4. embed the output batch
        tgt_sequences = self.tgt_embedings(tgt_sequences.type(LongTensor)).type(FloatTensor)

        # 4. decoder stack
        decoder_output = self.decoder(x=tgt_sequences, memory=encoder_output, self_mask=self_mask, memory_mask=None)

        # 5. classifier
        logits = self.classifier(decoder_output)

        return logits

    def save(self, model_dir: str, epoch_idx: int, loss_value: float) -> None:
        """
        Method to save a model along with a couple of information: number of training epochs and reached loss.

        # TODO: Could be extended if wish to save more statistics and state of model (e.g. 'converged' or not).

        :param model_dir: Directory where the model will be saved.

        :param epoch_idx: Epoch number.

        :param loss_value: Reached loss value at end of epoch ``epoch_idx``.
        """

        # Checkpoint to be saved.
        chkpt = {'name': 'Transformer',
                 'state_dict': self.state_dict(),
                 'model_timestamp': datetime.now(),
                 'epoch': epoch_idx,
                 'loss': loss_value
                 }

        filename = model_dir + 'model_epoch_{}.pt'.format(epoch_idx)
        torch.save(chkpt, filename)

    def load(self, checkpoint_file, logger) -> None:
        """
        Loads a model from the specified checkpoint file.

        :param checkpoint_file: File containing dictionary with model state and statistics.

        :param: logger: Logger object (to indicate number of trained epochs and loss value from loaded model).

        """
        # Load checkpoint
        # This is to be able to load a CUDA-trained model on CPU
        chkpt = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        # Load model.
        self.load_state_dict(chkpt['state_dict'])

        # Print statistics.
        logger.info(
            "Imported Transformer parameters from checkpoint from {} (epoch: {}, loss: {})".format(
                chkpt['model_timestamp'],
                chkpt['epoch'],
                chkpt['loss'],
                ))
