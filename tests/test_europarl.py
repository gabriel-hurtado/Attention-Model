from unittest import TestCase

import torch
import os

from dataset.europarl import Europarl, EuroparlLanguage, Split


class TestDecoderLayer(TestCase):
    def test_forward(self):
        dataset = Europarl(language=EuroparlLanguage.fr_en, split=Split.Train, split_size=0.6)
        source, target = dataset[0]


        # unit tests
        self.assertIsInstance(source, str)
        self.assertIsInstance(target, str)