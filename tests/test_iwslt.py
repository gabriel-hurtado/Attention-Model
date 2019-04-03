from os import getenv
from unittest import TestCase, skipIf

from dataset.iwslt import IWSLTDatasetBuilder, IWSLTLanguagePair
from dataset.utils import Split


class TestIWSLT(TestCase):
    @skipIf(len(getenv("CI", "")) > 0, "skipping slow tests on CI")
    def test_getitem(self):
        batch_size = 64
        language_pair = IWSLTLanguagePair.fr_en

        dataset_iterator = IWSLTDatasetBuilder.build(language_pair=language_pair,
                                                     split=Split.Train,
                                                     max_length=5, batch_size=batch_size)
        batch = next(iter(dataset_iterator))
