from os import getenv
from unittest import TestCase, skipIf

from dataset.iwslt import IWSLTDatasetBuilder
from dataset.language_pairs import LanguagePair
from dataset.utils import Split


class TestIWSLT(TestCase):
    @skipIf(len(getenv("CI", "")) > 0, "skipping slow tests on CI")
    def test_getitem(self):
        batch_size = 64
        language_pair = LanguagePair.fr_en

        dataset_iterator, val_iterator, _, _ = IWSLTDatasetBuilder.build(language_pair=language_pair,
                                                     split=Split.Train,
                                                     max_length=100, batch_size_train=batch_size)
        self.assertIsNotNone(dataset_iterator)
        self.assertIsNone(val_iterator)
        batch = next(iter(dataset_iterator))
