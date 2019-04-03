from enum import IntEnum, auto

from torch.utils import data
from torchtext import data, datasets

from dataset.europarl import Split
from dataset.language_pairs import LanguagePair
from dataset.utils import Tokenizer

ROOT_DATASET_DIR = "resources/torchtext"


class IWSLTDatasetBuilder():
    @staticmethod
    def build(language_pair: LanguagePair, split: Split, max_length=100, min_freq=2,
              start_token="<s>", eos_token="</s>", blank_token="<blank>", batch_size=32):
        """
        Initializes an iterator over the IWSLT dataset.
        The iterator then yields batches of size `batch_size`.

        Example:
        >>> dataset_iterator = IWSLTDatasetBuilder.build(language_pair=language_pair,
        ...                                              split=Split.Train,
        ...                                              max_length=5,
        ...                                              batch_size=batch_size)
        >>> batch = next(iter(dataset_iterator))

        :param language_pair: The language pair for which to create a vocabulary.
        :param split: The split type.
        :param max_length: Max length of sequence.
        :param min_freq: The minimum frequency a word should have to be included in the vocabulary
        :param start_token: The token that marks the beginning of a sequence.
        :param eos_token: The token that marks an end of sequence.
        :param blank_token: The token to pad with.
        """
        # load corresponding tokenizer
        source_tokenizer, target_tokenizer = language_pair.tokenizer()
        # create pytorchtext data field to generate vocabulary
        source_field = data.Field(tokenize=source_tokenizer, pad_token=blank_token)
        target_field = data.Field(tokenize=target_tokenizer, init_token=start_token,
                                  eos_token=eos_token, pad_token=blank_token)

        # Generates train and validation datasets
        # noinspection PyTypeChecker
        train, validation = datasets.IWSLT.splits(
            root=ROOT_DATASET_DIR,  # To check if the dataset was already downloaded
            exts=language_pair.extensions(),
            fields=(source_field, target_field),
            test=None,
            filter_pred=lambda x: (len(x.src), len(x.trg)) <= (max_length, max_length)
        )

        if split == Split.Train:
            dataset = train
        elif split == Split.Validation:
            dataset = validation
        else:
            raise NotImplementedError()

        source_field.build_vocab(train, min_freq=min_freq)  # Build vocabulary on training set
        target_field.build_vocab(train, min_freq=min_freq)
        return data.BucketIterator(
            dataset=dataset, batch_size=batch_size,
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg))
        )
