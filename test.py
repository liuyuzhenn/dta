import argparse
import os
import yaml

import sys
sys.path.append('.')
from src.utils import test as ttest
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='./configs/default.yml')
args = parser.parse_args()

with open(args.configs, 'r') as f:
    configs = yaml.full_load(f)

ttest(configs)
