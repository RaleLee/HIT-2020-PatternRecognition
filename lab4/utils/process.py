import torch
import torch.nn as nn
import torch.optim as optimize
from torch.autograd import Variable

import os
import time
import random
import numpy as np
from tqdm import tqdm


class Processor(object):

    def __init__(self, dataset, model, batch_size: int, args):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = batch_size
        self.__args = args

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()
            time_com = time.time() - time_start
            print('The model has been loaded into GPU and costs {:6f} s.\n'.format(time_com))

        self.__criterion = nn.NLLLoss()
        self.__optimizer = optimize.Adam(
            self.__model.parameters(), lr=self.__args.learning_rate,
            weight_decay=self.__args.l2_penalty
        )

    def train(self):
        best_train_acc = 0.0
        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(self.__args.epoch):
            total_loss = 0.0
            time_start = time.time()
            self.__model.train()

            for vector_batch, label_batch in tqdm(dataloader, ncols=50):
                vector_var = Variable(torch.FloatTensor(vector_batch))
                label_var = Variable(torch.LongTensor(label_batch))

                if torch.cuda.is_available():
                    vector_var = vector_var.cuda()
                    label_var = label_var.cuda()

                label_out = self.__model(vector_var)
                batch_loss = self.__criterion(label_out, label_var)
                self.__optimizer.zero_grad()
                batch_loss.backward()
                self.__optimizer.step()

                try:
                    total_loss += batch_loss.cpu().item()
                except AttributeError:
                    total_loss += batch_loss.cpu().data.numpy()[0]

                time_con = time.time() - time_start
                print('[Epoch {:2d}]: The total loss is {:2.6f}, cost {:2.6}s.'.format(epoch, batch_loss, time_con))

                change, time_start = False, time.time()
                cur_acc = self.estimate(is_train=True, batch_size=self.__batch_size)
                if cur_acc > best_train_acc:
                    test_acc = self.estimate(is_train=False, batch_size=self.__batch_size)
                    print('Test result: epoch: {}, label acc is {:.6f}.'.format(epoch, test_acc))

                torch.save(self.__model, os.path.join(self.__args.save_dir, 'model.pkl'))
                time_con = time.time() - time_start

                print('[Epoch {:2d}]: Test and save model cost {:2.6}s.'.format(epoch, time_con))

    def estimate(self, is_train=True, batch_size=100):
        if is_train:
            pred_label, real_label = self.prediction(
                self.__model, self.__dataset, 'train', batch_size
            )
        else:
            pred_label, real_label = self.prediction(
                self.__model, self.__dataset, 'test', batch_size
            )

        label_acc = accuracy(pred_label, real_label)
        return label_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()
        if mode == 'train':
            dataloader = dataset.batch_delivery('train', batch_size=batch_size, shuffle=False)
        elif mode == 'test':
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False)
        else:
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False)

        pred_label, real_label = [], []

        with torch.no_grad():
            for vector_batch, label_batch in tqdm(dataloader, ncols=50):
                real_label.append(label_batch)
                var_vector = Variable(torch.FloatTensor(vector_batch))
                if torch.cuda.is_available():
                    var_vector = var_vector.cuda()

                label_idx = model(var_vector, n_predicts=1)
                pred_label.append(label_idx)

        return pred_label, real_label


def expand_list(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            for sub_item in expand_list(item):
                yield sub_item
        else:
            yield item


def accuracy(pred, real):
    pred_array = np.array(list(expand_list(pred))).astype(np.uint8)
    real_array = np.array(list(expand_list(real))).astype(np.uint8)
    assert len(pred_array) == len(real_array)
    return (pred_array == real_array).sum() * 1.0 / len(pred_array)
