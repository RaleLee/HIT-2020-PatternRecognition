from utils.process import Processor
from utils.loader import DatasetManager

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', '-dd', type=str, default='data')
parser.add_argument('--save_dir', '-sd', type=str, default='save')
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

    dataset = DatasetManager(args)
    dataset.build_exam_dataset()

    Processor.test(os.path.join(args.save_dir, 'model.pkl'), dataset, args)
