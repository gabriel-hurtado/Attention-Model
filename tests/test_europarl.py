from unittest import TestCase

from dataset.europarl import Europarl, EuroparlLanguage, Split

class TestEuroparl(TestCase):
    def test_getitem(self):
        batch_size = 64
        dataset = Europarl(language=EuroparlLanguage.fr_en, split=Split.Train, split_size=0.6)
        source, target = dataset[0]

        # unit tests
        self.assertIsInstance(source, str)
        self.assertIsInstance(target, str)

