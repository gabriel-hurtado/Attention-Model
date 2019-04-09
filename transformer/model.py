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

    def forward(self, src_sequences, src_mask, trg_sequences, trg_mask) -> torch.Tensor:
        """
        Main forward pass of the model. Simplified worfklow:

        src_sequences -> Embeddings + Positional Encodings -> Encoder stack -> Memory --|
                                                                 |----------------------|
                                                                 v
        trg_sequences -> Embeddings + Positional Encodings -> Decoder stack -> Classifier -> Logits


        :param src_sequences: Batch of tokenized input sentences. Should be of shape (batch_size, in_seq_len).

        :param  src_mask: Mask, hiding the padding in the input batch. Should be same shape as src_sequences.

        :param trg_sequences: Batch of output sentences. Should be of shape (batch_size, out_seq_len).

        :param trg_mask: Mask, hiding the padding in the output batch. Should be the same shape as trg_sequences.

        .. note::

            This mask (which hides padding) will be combined with the `subsequent_mask` which hides subsequent
            positions in the decoder, to form only one mask.


        :return: Logits, of shape (batch_size, out_seq_len, d_model)
        """

        # 1. embed the input batch: have to move input sequences to torch.*.LongTensor
        src_sequences = self.src_embedings(src_sequences.type(LongTensor))

        # 2. encoder stack
        encoder_output = self.encoder(src=src_sequences, mask=src_mask, verbose=False)

        # 3. get subsequent mask to hide subsequent positions in the decoder.
        self_mask = subsequent_mask(trg_sequences.shape[1])

        # 3.5 Combine the trg_mask (which hides padding) and self_mask (which hide subsequent positions in the decoder)
        # as one mask

        hide_padding_and_future_words_mask = trg_mask.type_as(self_mask.data) & self_mask

        # 4. embed the output batch
        trg_sequences = self.tgt_embedings(trg_sequences.type(LongTensor))

        # 4. decoder stack
        decoder_output = self.decoder(x=trg_sequences, memory=encoder_output,
                                      self_mask=hide_padding_and_future_words_mask,
                                      memory_mask=src_mask)

        # 5. classifier
        logits = self.classifier(decoder_output)

        return logits

    def greedy_decode(self, src: torch.Tensor, src_mask: torch.Tensor, start_symbol=1) -> torch.Tensor:
        """
        Returns the prediction for `src` using greedy decoding for simplicity:

            - Feed `src` (after embedding) in the Encoder to get the "memory",
            - Feed an initial tensor (filled with start_symbol) in the Decoder, with the "memory" and the appropriate corresponding mask
            - Get the predictions of the model, makes a max to get the next token, cat it to the previous prediction and iterate


        :param src: sample for which to produce predictions.

        :param src_mask: Associated `src` mask

        :param start_symbol: Symbol used as initial value for the Decoder. Should correspond to start_token="<s>" in the translation task.

        """
        # 0. Ensure inference mode
        self.eval()

        # 1. Embed src
        embedded = self.src_embedings(src.type(LongTensor))

        # 2. Encode embedded inputs
        memory = self.encoder(src=embedded, mask=src_mask)

        # 3. Create initial input for decoder
        decoder_in = torch.ones(src.shape[0], 1).type(FloatTensor) * start_symbol

        for i in range(src.shape[1] - 1):

            # 4. Embed decoder_in
            decoder_in_embed = self.tgt_embedings(decoder_in.type(LongTensor))

            # 5. Go through decoder
            out = self.decoder(x=decoder_in_embed, memory=memory,
                               self_mask=subsequent_mask(decoder_in.shape[1]),
                               memory_mask=src_mask)

            # 6. classifier: TODO: Why only last word?
            logits = self.classifier(out[:, -1])

            # 7. Get predicted token for each sample in the batch
            _, next_token = logits.max(dim=1, keepdim=True)

            # 8. Concatenate predicted token with previous predictions
            decoder_in = torch.cat([decoder_in, next_token.type(FloatTensor)], dim=1)

        # 9. Return entire prediction
        return decoder_in

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
