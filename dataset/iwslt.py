from enum import IntEnum, auto
from dataset.utils import Tokenizer
from torch.utils import data
from dataset.europarl import Split
from torchtext import data, datasets


class IWSLTLanguage(IntEnum):
    fr_en = auto()

    def tokenizer(self):
        if self == IWSLTLanguage.fr_en:
            return (
                Tokenizer(language='fr'),
                Tokenizer(language='en')
            )

    def extensions(self):
        if self == IWSLTLanguage.fr_en:
            return (
                '.fr',
                '.en'
            )

    def path(self):
        if self == IWSLTLanguage.fr_en:
            return ".data/iwslt/fr-en/"


class IWSLT(data.Dataset):
    def __init__(self, language: IWSLTLanguage, split: Split, split_ratio=0.6, random_state=100):

        # load corresponding tokenizer
        source_tokenizer, target_tokenizer = language.tokenizer()
        # load corresponding extensions for IWSLT loading
        source_extension, target_extension = language.extensions()
        # path to dataset (if already downloaded)
        self.path = language.path()
        # tokenize for beginning, end of sequence and blank
        b_of_sequence_word = '<s>'
        e_of_sequence_word = '</s>'
        blank_word = "<blank>"

        # create pytorchtext data field to generate vocabulary
        self.source_field = data.Field(tokenize=source_tokenizer, pad_token=blank_word)
        self.target_field = data.Field(tokenize=target_tokenizer, init_token=b_of_sequence_word,
                                       eos_token=e_of_sequence_word, pad_token=blank_word)

        # max length of sequence
        max_length = 100

        # split dataset by loading corresponding source and target files from .data/ folder
        train, validation, test = datasets.IWSLT.splits(
            split_ratio=split_ratio,
            random_state=random_state,
            check=self.path,# check if dataset already downloaded
            exts=(source_extension, target_extension),
            fields=(self.source_field, self.target_field),
            filter_pred=lambda x: len(vars(x)['src']) <= max_length and len(vars(x)['trg']) <= max_length)

        if split == Split.Train:
            self.array = train
        elif split == Split.Validation:
            self.array = validation
        else:
            raise NotImplementedError()

    def build_vocabulary(self):
        min_freq = 2
        # build source and target vocabulary for all words > MIN_FREQ
        # take some times
        #self.source_field.build_vocab(train.src, min_freq=min_freq)
        #self.target_field.build_vocab(train.trg, min_freq=min_freq)

    def __getitem__(self, index):
        source, target = self.array[index]
        print(source)
        return source, target

    def __len__(self):
        return len(self.indexes)
