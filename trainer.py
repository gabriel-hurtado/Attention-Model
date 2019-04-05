import torch
from transformer.model import Transformer
from training.optimizer import NoamOpt
from training.loss import LabelSmoothingLoss


class Trainer(object):
    """
    Represents a worker taking care of the training of an instance of the ``Transformer`` model.

    """

    def __init__(self, params):
        """
        Constructor of the Trainer.
        Sets up the following:
            - Device available (e.g. if CUDA is present)
            - Initialize the model, dataset, loss, optimizer
            - log statistics (epoch, elapsed time, BLEU score etc.)
        """

        # if CUDA available, moves computations to GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # instantiate model
        model = Transformer(params["model"]).to(device)

        # instantiate loss
        # TODO: hardcode 0 as padding token for now, verify with Dataset class later
        loss = LabelSmoothingLoss(size=params["model"]["tgt_vocab_size"], padding_token=0, smoothing=0.1)

        # instantiate optimizer
        optimizer = NoamOpt(model=model, model_size=params["model"]["d_model"], factor=2, warmup=4000)

        # missing dataset class

        # missing dataloader

        # missing logger

    def train(self):
        pass

