import argparse
import numpy as np
import os
import yaml

import torch
import sys
sys.path.append('.')
from src.utils import train as ttrain

torch.set_num_threads(8)

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='configs/scannet/default.yml')
args = parser.parse_args()

with open(args.configs, 'r') as f:
    configs = yaml.full_load(f)

seed = configs.get('seed',0)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

ttrain(configs)
