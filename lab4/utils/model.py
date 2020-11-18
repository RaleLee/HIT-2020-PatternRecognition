import torch.nn as nn
import torch.nn.functional as F


class ModelManager(nn.Module):

    def __init__(self, args, num_label):
        super(ModelManager, self).__init__()
        self.__args = args
        self.__num_label = num_label
        # encoder
        self.__encoder_dropout = nn.Dropout(self.__args.dropout_rate)
        self.__encoder = nn.Linear(self.__args.embedding_dim, self.__args.hidden_dim)

        # hidden layer
        self.__hidden_layer = nn.Linear(self.__args.hidden_dim, self.__args.hidden_dim // 2)
        self.__norm = nn.BatchNorm1d(self.__args.hidden_dim // 2)
        self.__relu = nn.ReLU()

        # decoder
        self.__decoder_dropout = nn.Dropout(self.__args.dropout_rate)
        self.__decoder = nn.Linear(self.__args.hidden_dim // 2, num_label)

    def forward(self, vector, n_predicts=None):
        drop_vector = self.__encoder_dropout(vector)
        encoded_vector = self.__encoder(drop_vector)

        x = self.__hidden_layer(encoded_vector)
        x = self.__norm(x)
        x = self.__relu(x)

        x = self.__decoder_dropout(x)
        pred = self.__decoder(x)
        if n_predicts is None:
            return F.log_softmax(pred, dim=-1)
        else:
            _, pred_index = pred.topk(n_predicts, dim=-1)
            return pred_index.cpu().data.numpy().tolist()
