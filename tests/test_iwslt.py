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

<<<<<<< HEAD
        dataset = IWSLT(language_pair=language, split=Split.Train, split_ratio=0.6, max_length=5)
        source, target = dataset[0]

        # unit tests
        self.assertIsInstance(source, str)
        self.assertIsInstance(target, str)
=======
        dataset_iterator = IWSLTDatasetBuilder.build(language_pair=language_pair,
                                                     split=Split.Train,
                                                     max_length=100, batch_size=batch_size)
        batch = next(iter(dataset_iterator))
>>>>>>> a14a30ef4900be83a9343de00629ce37c714cacf
