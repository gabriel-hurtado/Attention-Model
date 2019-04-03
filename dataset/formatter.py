import numpy as np
from torch import Tensor
from torch.autograd import Variable

from dataset.europarl import Europarl, Split
from dataset.language_pairs import LanguagePair
from transformer.utils import subsequent_mask


class BatchWrapper:
    """Holds a batch of data that it can partially mask during training."""

    def __init__(self, source, target=None, pad=0):
        # save source and mask for use during training
        self.source = source
        self.source_mask = (source != pad).unsqueeze(-2)

        if target is not None:
            self.target = target[:, :-1]
            # shift one position to the right so the target is the next word
            self.target_ = target[:, 1:]
            # create mask to hide padding and future words (subsequent)
            self.target_mask = self.make_std_mask(self.target, pad)
            # investigate what ntokens stands for ?
            # self.ntokens = (self.target_ != pad).data.sum()

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
