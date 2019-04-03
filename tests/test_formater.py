from os import getenv
from unittest import TestCase, skipIf

from torch.utils.data import DataLoader

from dataset.europarl import EuroparlLanguage, Split
from dataset.formater import Formater


class TestFormater(TestCase):
    @skipIf(len(getenv("CI", "")) > 0, "skipping slow tests on CI")
    def test_collate_fn(self):
        # initialization
        batch_size = 2
        formater_dataset = Formater(language=EuroparlLanguage.fr_en, split=Split.Train,
                                    split_size=0.6)

        # instantiate DataLoader object
        dataloader = DataLoader(formater_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=formater_dataset.collate_fn, num_workers=0,
                                sampler=None)
