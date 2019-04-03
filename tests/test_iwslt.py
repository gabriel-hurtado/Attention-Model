from unittest import TestCase

from dataset.iwslt import IWSLT, IWSLTLanguage
from dataset.utils import Split

class TestIWSLT(TestCase):
    def test_getitem(self):
        batch_size = 64

        source, targ = IWSLTLanguage.fr_en.tokenizer()
        dataset = IWSLT(language=IWSLTLanguage.fr_en, split=Split.Train, split_ratio=0.6)
        source, target = dataset[0]

        # unit tests
        self.assertIsInstance(source, str)
        self.assertIsInstance(target, str)

