import numpy as np
from torch import Tensor
from torch.autograd import Variable
from torchtext.data import Batch, Dataset, Field

from dataset.europarl import Europarl, Split
from dataset.language_pairs import LanguagePair
from transformer.utils import subsequent_mask


class BatchMasker(Batch):
    """
    Handles the masking in a given batch.
    """

    def __init__(self, batch: Batch, padding_token: str = "<blank>"):
        """
        Constructs a batch masker.
        The batch must have two fields: 'src' and 'trg', respectively the
        source and target masks.

        :param batch: The batch to mask out.
        :param padding_token: The token used to pad that must be
        """
        super().__init__()
        self.batch = batch

        self.src_field = self.dataset.fields['src']  # type: Field
        self.trg_field = self.dataset.fields['trg']  # type: Field

        # Find integer value of "padding" in the respective vocabularies
        src_padding = self.src_field.vocab.stoi[padding_token]
        trg_padding = self.trg_field.vocab.stoi[padding_token]

        masker = BatchWrapper(source=self.src, target=self.trg,
                              pad_src=src_padding, pad_trg=trg_padding)
        # The masks are everywhere the tensor is not equal to the padding value
        # Additionnaly, we mask
        self.src_mask = masker.source_mask  # type: Tensor
        self.trg_mask = masker.target_mask  # type: Tensor

    @property
    def batch_size(self):
        return self.batch.batch_size

    @property
    def src(self) -> Tensor:
        return self.batch.src

    @property
    def trg(self) -> Tensor:
        return self.batch.trg

    @property
    def dataset(self) -> Dataset:
        return self.batch.dataset

    @property
    def fields(self):
        return self.batch.fields

    @property
    def input_fields(self):
        return self.batch.input_fields

    @property
    def target_fields(self):
        return self.batch.target_fields


class BatchWrapper:
    """Holds a batch of data that it can partially mask during training."""

    def __init__(self, source: Tensor, target: Tensor = None, pad_src=0, pad_trg=0):
        # save source and mask for use during training
        self.source = source
        self.source_mask = (source != pad_src)  # type: Tensor
        # Adds a dimension in the middle (equivalent to vec = vec[:,None,:])
        self.source_mask.unsqueeze_(-2)

        if target is not None:
            self.target = target[:, :-1]
            # create mask to hide padding and future words (subsequent)
            self.target_mask = self.make_std_mask(self.target, pad_trg)
            # ntokens is the size of the sentence (excluding padding)
            self.ntokens = (target[:, 1:] != pad_trg).data.sum()

    @staticmethod
    def make_std_mask(target, pad):
        """Creates a mask to hide padding and future words."""
        # hide padding
        target_mask = (target != pad).unsqueeze(-2)
        # hide padding and future words
        target_mask = target_mask & Variable(
            subsequent_mask(target.size(-1)).type_as(target_mask.data))
        return target_mask


class Formater(Europarl):
    def __init__(self, language: LanguagePair, split: Split, split_size=0.6, pad=0):
        # call base constructor
        super(Formater, self).__init__(language, split, split_size)
        self.pad = pad

    def __getitem__(self, index):
        # call Europarl get item
        source, target = super(Formater, self).__getitem__(index)
        # tokenize the source and target
        return [self.source_tokenizer(source), self.target_tokenizer(target)]

    def collate_fn(self, batch):
        # Get batch-wise max sequence length
        max_seq_length = np.max([max(len(source), len(target)) for [source, target] in batch])

        # Pad to the max sequence length
        # TODO: Pad the entire batch once with np.pad() if size of pad is diff for each seq
        padded = np.array([[np.pad(arr, (0, max_seq_length - len(arr)), mode='constant',
                                   constant_values=self.pad)
                            for arr in pair] for pair in batch])

        source, target = padded[:, 0, :], padded[:, 1, :]
        return BatchWrapper(Tensor(source), Tensor(target), pad=self.pad)

    def __len__(self):
        return len(self.indexes)
