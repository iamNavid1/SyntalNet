from __future__ import annotations

import os
import logging
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter


def setup_logger(log_dir: str, name: str = "run") -> Tuple[logging.Logger, SummaryWriter]:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    writer = SummaryWriter(log_dir)
    return logger, writer
