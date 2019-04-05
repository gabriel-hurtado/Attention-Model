import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CopyTaskDataset(Dataset):
    """
    A simple copy task: The network is asked to produce outputs similar to the inputs.

    This is used as a basic test to ensure gradients are flowing correctly through the network, and the latter
    is able to overfit a small amount of data.
    """
    def __init__(self, max_int: int, max_seq_length: int, size: int,):
        """
        Constructor of the ``CopyTaskDataset``.

        :param max_int: Upper bound on the randomly drawn samples.

        :param max_seq_length: Sequence length. Will be the same for all samples.

        :param size: Size of the dataset. Mainly used for ``__len__`` as generated samples are random (and bounded by ``max_len``).
        """
        self.max_int = max_int
        self.max_seq_length = max_seq_length
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        """
        Randomly creates a sample of shape [1, self.max_seq_length], where the elements are drawn randomly
        in U[0, self.max_int].

        As this is a copy task, inputs = targets.

        :param item: index of the sample, not used here.

        :return: tuple (inputs, targets) of identical shape.
        """

        sample = torch.randint(low=0, high=self.max_int, size=(self.max_seq_length,), device=device)

        return sample, sample

    def collate(self, samples):
        inputs, targets = default_collate(samples)

        return Batch(inputs, targets)


class Batch:
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences

    def cuda(self):
        self.src_sequences = self.src_sequences.cuda()
        self.tgt_sequences = self.tgt_sequences.cuda()
