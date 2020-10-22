import myutils
from config import get_config
from learner import metric_learner
import argparse
from pathlib import Path
import numpy as np
import torch

if __name__ == '__main__':

    conf = get_config()

    learner = metric_learner(conf)

    learner.load_bninception_pretrained(conf)

    learner.train(conf)

