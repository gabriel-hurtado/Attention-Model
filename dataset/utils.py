import spacy
from enum import IntEnum, auto

class Tokenizer(object):
    """ Tokenizer
     Tokenizes the text, i.e. segments it into words, punctuation and so on.
     This is done by applying rules specific to each language.
     For example, punctuation at the end of a sentence should be split off – whereas “U.K.” should remain one token.
     see doc at https://spacy.io/usage/spacy-101"""

    def __init__(self, language='en'):
        """
        constructor, load appropriate model from spacy api
        :param language: model string id
        """
        self.nlp = spacy.load(language)

    def __call__(self, text):
        """
        tokenize a string in corresponding token
        :param text: string to be tokenized
        :return: list of tokens
        """
        return [tok.text for tok in self.nlp.tokenizer(text)]



class Split(IntEnum):
    Train = auto()
    Validation = auto()
