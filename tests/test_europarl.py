from unittest import TestCase

import torch
import numpy as np
from torch.autograd import Variable

from dataset.europarl import Europarl, EuroparlLanguage, Split, BatchWrapper


class TestDecoderLayer(TestCase):
    def test_forward(self):
        dataset = Europarl(language=EuroparlLanguage.fr_en, split=Split.Train, split_size=0.6)
        source, target = dataset[0]

        # unit tests
        self.assertIsInstance(source, str)
        self.assertIsInstance(target, str)

        # mimic DataLoader
        batch_size, batch_source, batch_target = 2, [], []
        for i in range(batch_size):
            source, target = dataset[i]
            batch_source.append(list(source))
            batch_target.append(list(target))


        data = torch.from_numpy(np.random.randint(1, 8, size=(batch_size, 10)))
        data[:, 0] = 1
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        batch_wrap = BatchWrapper(source, target, 0)

        print(batch_wrap)
