from utils.process import Processor
from utils.loader import DatasetManager
from utils.model import ModelManager
import torch
import os
import platform
import argparse
import random
import json
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', '-dd', type=str, default='data')
parser.add_argument('--save_dir', '-sd', type=str, default='save')
# parser.add_argument('--log_dir', '-ld', type=str, default=)
parser.add_argument('--epoch', '-e', type=int, default=100)
parser.add_argument('--random_state', '-rs', type=int, default=2020)
parser.add_argument('--embedding_dim', '-ed', type=int, default=85)
parser.add_argument('--hidden_dim', '-hd', type=int, default=256)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.1)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
parser.add_argument('--batch_size', '-bs', type=int, default=32)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-5)

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        if platform.platform().find('Windows') >= 0:
            os.system("mkdir " + args.save_dir)
        else:
            os.system("mkdir -p " + args.save_dir)

    param_file_path = os.path.join(args.save_dir, 'param.json')
    with open(param_file_path, 'w') as f:
        f.write(json.dumps(args.__dict__, indent=True))

    # Fix the random state
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)

    # Fix for CPU
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    dataset = DatasetManager(args)
    dataset.build_dataset()

    model = ModelManager(args, 10)
    process = Processor(dataset, model, args.batch_size, args)
    process.train()

    print("Training Over")
    Processor.test(os.path.join(args.save_dir, 'model.pkl'), dataset, args)
