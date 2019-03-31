from enum import IntEnum, auto

import numpy as np
from torch.utils import data
from torch.autograd import Variable
from transformer.utils import  subsequent_mask

class Split(IntEnum):
    Train = auto()
    Validation = auto()


class BatchWrapper:
    "Object for holding a batch of data with mask during training."

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
            self.ntokens = (self.target_ != pad).data.sum()

    @staticmethod
    def make_std_mask(target, pad):
        "Create a mask to hide padding and future words."
        # hide padding
        target_mask = (target != pad).unsqueeze(-2)
        # hide padding and future words
        target_mask = target_mask & Variable(subsequent_mask(target.size(-1)).type_as(target_mask.data))
        return target_mask


class EuroparlLanguage(IntEnum):
    fr_en = auto()

    def path(self):
        if self == EuroparlLanguage.fr_en:
            return (
                "resources/europarl/fr-en/europarl-v7.fr-en.fr.big.txt",
                "resources/europarl/fr-en/europarl-v7.fr-en.en.big.txt"
            )


class Europarl(data.Dataset):
    def __init__(self, language: EuroparlLanguage, split: Split, split_size=0.6):
        self.language = language
        self.split = split
        # Load file in memory
        path_lang1, path_lang2 = language.path()
        with open(path_lang1) as f1, open(path_lang2) as f2:
            self.array = [tuple(l.strip() for l in lines) for lines in zip(f1, f2)]

        # Train/Val split
        n = len(self.array)
        self.indexes = np.arange(n)
        np.random.shuffle(self.indexes)

        train_size = round(n * split_size)
        if split == Split.Train:
            self.indexes = self.indexes[:train_size]
        elif split == Split.Validation:
            self.indexes = self.indexes[train_size:]
        else:
            raise NotImplementedError()


    def __getitem__(self, index):
        source, target = self.array[self.indexes[index]]
        return source, target

    def content_len(self, batch_source, batch_target):
        """
        Called via the DataLoader during post processing of batch of data.
        Wrap the batch in
        :param batch: batch of sample source and target data. (Batch_size x Sequence x Dimension)
        :return: BatchWrapper object with batches of source, target and masks.
        """
        batch_wrapper = BatchWrapper(batch_source, batch_target, pad=0)

        return batch_wrapper

    def __len__(self):
        return len(self.indexes)
