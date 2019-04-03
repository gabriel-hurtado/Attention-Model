from unittest import TestCase

from torch.utils.data import DataLoader

from dataset.europarl import Europarl, EuroparlLanguage, Split
from dataset.formater import Formater, BatchWrapper


class TestFormater(TestCase):

    """
    def test_getitem(self):
        formater_dataset = Formater(language=EuroparlLanguage.fr_en, split=Split.Train, split_size=0.6)
        source, target = formater_dataset[0]

        # unit tests
        self.assertIsInstance(source, list, msg=f'formatted source sample is not a list')
        self.assertIsInstance(target, list, msg=f'formatted target sample is not a list')
    """
    def test_collate_fn(self):
        # initialization
        batch_size = 2
        formater_dataset = Formater(language=EuroparlLanguage.fr_en, split=Split.Train, split_size=0.6)

        # instantiate DataLoader object
        dataloader = DataLoader(formater_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=formater_dataset.collate_fn, num_workers=0, sampler=None)

        for i, batch in enumerate(dataloader):
            print(i, batch)
            break