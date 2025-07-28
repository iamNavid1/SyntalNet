import os
import logging
from torch.utils.tensorboard import SummaryWriter


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'run.log')),
            logging.StreamHandler()
        ]
    )
    writer = SummaryWriter(log_dir)
    return logging.getLogger(), writer