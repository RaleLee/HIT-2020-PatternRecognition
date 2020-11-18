import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TorchDataset(Dataset):

    def __init__(self, vector, label):
        self.__vector = vector
        self.__label = label

    def __getitem__(self, index):
        return self.__vector[index], self.__label[index]

    def __len__(self):
        assert len(self.__vector) == len(self.__label)
        return len(self.__vector)


class DatasetManager(object):

    def __init__(self, args):
        self.__args = args
        self.__vector_data = {}
        self.__label_data = {}

    def build_dataset(self):
        vector_path = os.path.join(self.__args.data_dir, 'TrainSamples.csv')
        label_path = os.path.join(self.__args.data_dir, 'TrainLabels.csv')
        all_vector = np.genfromtxt(vector_path, delimiter=',')
        all_label = np.genfromtxt(label_path, delimiter='\n', dtype=np.int8)
        all_label = all_label.reshape((len(all_label), 1))

        train_data, test_data, train_label, test_label = train_test_split(all_vector, all_label, test_size=0.2,
                                                                          random_state=self.__args.random_state)

        self.__vector_data['train'] = train_data
        self.__label_data['train'] = train_label
        self.__vector_data['test'] = test_data
        self.__label_data['test'] = test_label

    def build_exam_dataset(self):
        vector_path = os.path.join(self.__args.data_dir, 'TestSamples.csv')
        all_vector = np.genfromtxt(vector_path, delimiter=',')
        fake_label = np.zeros((all_vector.shape[0], 1))
        self.__vector_data['exam'] = all_vector
        self.__label_data['exam'] = fake_label

    def batch_delivery(self, data_name, batch_size=None, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        vector = self.__vector_data[data_name]
        label = self.__label_data[data_name]
        dataset = TorchDataset(vector, label)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)

    @property
    def batch_size(self):
        return self.__args.batch_size

    @staticmethod
    def __collate_fn(batch):
        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch
