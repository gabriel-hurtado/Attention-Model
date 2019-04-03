from enum import IntEnum, auto

from torch.utils import data
from torchtext import data, datasets

from dataset.europarl import Split
from dataset.utils import Tokenizer

ROOT_DATASET_DIR="resources/torchtext"


class IWSLTLanguagePair(IntEnum):
    fr_en = auto()

    def tokenizer(self):
        if self == IWSLTLanguagePair.fr_en:
            return (
                Tokenizer(language='fr'),
                Tokenizer(language='en'),
            )
        else:
            raise ValueError()

    def extensions(self):
        if self == IWSLTLanguagePair.fr_en:
            return ('.fr', '.en')
        else:
            raise ValueError()


class IWSLT(data.Dataset):
    def __init__(self, language_pair: IWSLTLanguagePair, split: Split, split_ratio=0.6,
                 random_state=100, start_token="<s>", eos_token="</s>",
                 blank_token="<blank>", max_length=100):
        """
        Initializes an IWSLT dataset.

        :param language_pair: The language pair for which to create a vocabulary.
        :param split: The split type.
        :param split_ratio: The size of the split
        :param random_state: The random seed to use.
        :param start_token: The token that marks the beginning of a sequence.
        :param eos_token: The token that marks an end of sequence.
        :param blank_token: The token to pad with.
        :param max_length: Max length of sequence.
        """
        # load corresponding tokenizer
        source_tokenizer, target_tokenizer = language_pair.tokenizer()
        # create pytorchtext data field to generate vocabulary
        self.source_field = data.Field(tokenize=source_tokenizer, pad_token=blank_token)
        self.target_field = data.Field(tokenize=target_tokenizer, init_token=start_token,
                                       eos_token=eos_token, pad_token=blank_token)

        # split dataset by loading corresponding source and target files from .data/ folder
        train, validation, test = datasets.IWSLT.splits(
            split_ratio=split_ratio,
            random_state=random_state,
            root=ROOT_DATASET_DIR,  # To check if dataset already downloaded
            exts=language_pair.extensions(),
            fields=(self.source_field, self.target_field),
            filter_pred=lambda x: len(vars(x)['src']) <= max_length
                                  and len(vars(x)['trg']) <= max_length
        )

        if split == Split.Train:
            examples = train
        elif split == Split.Validation:
            examples = validation
        else:
            raise NotImplementedError()
        super().__init__(examples=examples,
                         fields=[("source", self.source_field), ("target", self.target_field)])

    def build_vocabulary(self):
        min_freq = 2
        # build source and target vocabulary for all words > MIN_FREQ
        # take some times
        #self.source_field.build_vocab(train.src, min_freq=min_freq)
        #self.target_field.build_vocab(train.trg, min_freq=min_freq)
