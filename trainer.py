import os
import json
import torch
import logging
import logging.config
from datetime import datetime
from torch.utils.data import DataLoader

from transformer.model import Transformer
from training.optimizer import NoamOpt
from training.loss import LabelSmoothingLoss, CrossEntropyLoss
from dataset.copy_task import CopyTaskDataset


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

        # configure all logging
        self.configure_logging(training_problem_name="copy_task", params=params)

        # instantiate model
        self.model = Transformer(params["model"]).to(device)

        # instantiate loss
        # TODO: hardcode -1 as padding token for now, verify with Dataset class later

        if "smoothing" in params["training"]:
            self.loss_fn = LabelSmoothingLoss(size=params["model"]["tgt_vocab_size"],
                                              padding_token=params["dataset"]["pad_token"],
                                              smoothing=params["training"]["smoothing"])
            self.logger.info("Using LabelSmoothingLoss with smoothing={}.".format(params["training"]["smoothing"]))
        else:
            self.loss_fn = CrossEntropyLoss(pad_token=params["dataset"]["pad_token"])
            self.logger.info("Using CrossEntropyLoss.")

        # instantiate optimizer
        self.optimizer = NoamOpt(model=self.model,
                                 model_size=params["model"]["d_model"],
                                 factor=params["optim"]["factor"],
                                 warmup=params["optim"]["warmup"])

        # get number of epochs and related hyper parameters
        self.epochs = params["training"]["epochs"]

        # initialize training Dataset class
        self.training_dataset = CopyTaskDataset(max_int=params["dataset"]["training"]["max_int"],
                                                max_seq_length=params["dataset"]["training"]["max_seq_length"],
                                                size=params["dataset"]["training"]["size"])

        # initialize DataLoader
        self.training_dataloader = DataLoader(dataset=self.training_dataset,
                                              batch_size=params["training"]["batch_size"],
                                              shuffle=False, num_workers=0,
                                              collate_fn=self.training_dataset.collate)

        # initialize validation Dataset class
        self.validation_dataset = CopyTaskDataset(max_int=params["dataset"]["validation"]["max_int"],
                                                  max_seq_length=params["dataset"]["validation"]["max_seq_length"],
                                                  size=params["dataset"]["validation"]["size"])

        # initialize Validation DataLoader
        self.validation_dataloader = DataLoader(dataset=self.validation_dataset,
                                                batch_size=len(self.validation_dataset),
                                                shuffle=False, num_workers=0,
                                                collate_fn=self.validation_dataset.collate)

        self.logger.info('Experiment setup done.')

    def train(self):
        """
        Main training loop.

            - Trains the Transformer model on the specified dataset for a given number of epochs
            - Logs statistics to logger for every batch per epoch

        """
        for epoch in range(self.epochs):

            self.model.train()
            for i, batch in enumerate(self.training_dataloader):

                # 1. reset all gradients
                self.optimizer.zero_grad()

                # Convert batch to CUDA.
                if torch.cuda.is_available():
                    batch.cuda()

                # 2. Perform forward calculation.
                logits = self.model(batch.src_sequences, None, batch.tgt_sequences, None)

                # 3. Evaluate loss function.
                loss = self.loss_fn(logits, batch.tgt_sequences)

                # 4. Backward gradient flow.
                loss.backward()

                # Log "elementary" statistics - episode and loss.
                self.logger.info('Epoch: {} | Episode: {} | Loss: {}'.format(epoch, i, loss.item()))

                # 5. Perform optimization.
                self.optimizer.step()

            # save model at end of each epoch
            self.model.save(self.model_dir, epoch, loss.item())
            self.logger.info("Model exported to checkpoint.")

            # validate the model on the validation set
            self.model.eval()
            for batch in self.validation_dataloader:

                # Convert batch to CUDA.
                if torch.cuda.is_available():
                    batch.cuda()

                # 1. Perform forward calculation.
                logits = self.model(batch.src_sequences, None, batch.tgt_sequences, None)

                # 2. Evaluate loss function.
                loss = self.loss_fn(logits, batch.tgt_sequences)

                # Log "elementary" statistics - episode and loss.
                self.logger.info('Validation Set | Loss: {}'.format(loss.item()))

    def configure_logging(self, training_problem_name: str, params: dict) -> None:
        """
        Takes care of the initialization of logging-related objects:

            - Sets up a logger with a specific configuration,
            - Sets up a logging directory
            - sets up a logging file in the log directory
            - Sets up a folder to store trained models

        :param training_problem_name: Name of the dataset / training task (e.g. "copy task", "IWLST"). Used for the logging
        folder name.
        """
        # instantiate logger
        # Load the default logger configuration.
        logger_config = {'version': 1,
                         'disable_existing_loggers': False,
                         'formatters': {
                             'simple': {
                                 'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                 'datefmt': '%Y-%m-%d %H:%M:%S'}},
                         'handlers': {
                             'console': {
                                 'class': 'logging.StreamHandler',
                                 'level': 'INFO',
                                 'formatter': 'simple',
                                 'stream': 'ext://sys.stdout'}},
                         'root': {'level': 'DEBUG',
                                  'handlers': ['console']}}

        logging.config.dictConfig(logger_config)

        # Create the Logger, set its label and logging level.
        self.logger = logging.getLogger(name='Trainer')

        # Prepare the output path for logging
        time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
        self.log_dir = 'experiments/' + training_problem_name + '/' + time_str + '/'

        os.makedirs(self.log_dir, exist_ok=False)
        self.logger.info('Folder {} created.'.format(self.log_dir))

        # Set log dir and add the handler for the logfile to the logger.
        self.log_file = self.log_dir + 'training.log'
        self.add_file_handler_to_logger(self.log_file)

        self.logger.info('Log File {} created.'.format(self.log_file))

        # Models dir: to store the trained models.
        self.model_dir = self.log_dir + 'models/'
        os.makedirs(self.model_dir, exist_ok=False)

        self.logger.info('Model folder {} created.'.format(self.model_dir))

        # save the configuration as a json file in the experiments dir
        with open(self.log_dir + 'params.json', 'w') as fp:
            json.dump(params, fp)

        self.logger.info('Configuration saved to {}.'.format(self.log_dir + 'params.json'))

    def add_file_handler_to_logger(self, logfile: str) -> None:
        """
        Add a ``logging.FileHandler`` to the logger.

        Specifies a ``logging.Formatter``:
            >>> logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
            >>>                   datefmt='%Y-%m-%d %H:%M:%S')

        :param logfile: File used by the ``FileHandler``.

        """
        # create file handler which logs even DEBUG messages
        fh = logging.FileHandler(logfile)

        # set logging level for this file
        fh.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        # add the handler to the logger
        self.logger.addHandler(fh)


if __name__ == '__main__':

    params = {
        "training": {
                "epochs": 5,
                "batch_size": 2,
                "smoothing": 0.1,
        },

        "optim": {
            "factor": 2,
            "warmup": 400

        },

        "dataset": {
            "pad_token": -1,

            'training': {
                    'max_int': 10,
                    'max_seq_length': 10,
                    'size': 10000},

            'validation': {
                'max_int': 10,
                'max_seq_length': 10,
                'size': 1000}

        },

        "model": {
                'd_model': 512,
                'src_vocab_size': 10,
                'tgt_vocab_size': 10,

                'N': 6,
                'dropout': 0.1,

                'attention': {'n_head': 8,
                              'd_k': 64,
                              'd_v': 64,
                              'dropout': 0.1},

                'feed-forward': {'d_ff': 2048,
                                 'dropout': 0.1}
        }
    }
    trainer = Trainer(params)
    trainer.train()

    src = torch.Tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    predictions = trainer.model.greedy_decode(src, None, start_symbol=1)
    print(predictions)
