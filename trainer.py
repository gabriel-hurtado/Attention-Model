import os
import torch
import logging
import logging.config
from datetime import datetime

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

        # configure all logging
        self.configure_logging(training_problem_name="copy_task")

        self.logger.info('Experiment setup done.')

    def train(self):
        pass

    def configure_logging(self, training_problem_name) -> None:
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

        # create folder
        os.makedirs(self.log_dir, exist_ok=False)

        # Set log dir and add the handler for the logfile to the logger.
        self.log_file = self.log_dir + 'training.log'
        self.add_file_handler_to_logger(self.log_file)

        # Models dir: to store the trained models.
        self.model_dir = self.log_dir + 'models/'
        os.makedirs(self.model_dir, exist_ok=False)

    def add_file_handler_to_logger(self, logfile):
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

