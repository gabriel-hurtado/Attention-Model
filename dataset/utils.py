from enum import IntEnum, auto

import spacy


class Tokenizer(object):
    """
    Tokenizes the text, i.e. segments it into words, punctuation and so on.
    This is done by applying rules specific to each language.
    For example, punctuation at the end of a sentence should be split off
        – whereas “U.K.” should remain one token.

    Documentation @ https://spacy.io/usage/spacy-101
    """

    def __init__(self, language: str):
        """
        Loads the appropriate model from Spacy's API

        :param language: model string id
        """
        self.nlp = spacy.load(language)

    def __call__(self, text: str):
        """
        tokenize a string in corresponding token
        :param text: String to be tokenized.
        :return: List of tokens.
        """
        return [tok.text for tok in self.nlp.tokenizer(text)]


class Split(IntEnum):
    Train = auto()
    Validation = auto()
