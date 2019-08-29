import torch
import numpy as np
import warnings
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import fire

class Configuration(object):
    def __init__(self):
        # data
        self.data_path = './nasdaq100_padding.csv'

        # hyperparameters
        self.epochs = 5000
        self.lr = 0.00005
        self.batch_size = 64
        self.weight_decay = 0.00005
        self.dropout = 0.1
        self.method = 'adam'
        self.encoder_hidden_size = 128
        self.decoder_hidden_size = 128
        self.conv_size = 128
        self.kernel_size = 2
        self.dropout=0.1

        # model
        self.T = 10
        self.use_gpu = True
        self.input_size = 81    # for nasdaq data
        self.num_workers = 4

    def _setting(self, kwargs=None):
        if kwargs:
            for k, v in kwargs.items():
                if not hasattr(self, k):
                    warnings.warn('warning: Configuration has no attribute %s' % str(k))
                setattr(self, k, v)
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self._print()

    def _print(self):
        print('-'*10, '>'*3, 'User Config', '<'*3, '-'*10)
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                print(k, '=', v)
        print('-'*37)


class NasdaqDataset(Data.Dataset):
    def __init__(self, opt, is_train=True,):
        self.T = opt.T
        dat = pd.read_csv(opt.data_path)
        self.X = self.preprocess_X_data(dat.loc[:, [x for x in dat.columns.tolist() if x != 'NDX']])
        # self.X = dat.loc[:, [x for x in dat.columns.tolist() if x != 'NDX']].values
        # self.y, self.y_mean, self.y_std = self.preprocess_y_data(np.array(dat.NDX))
        self.scale = np.max(np.array(dat.NDX))
        self.y, self.y_mean, self.y_std = self.preprocess_y_data(np.array(dat.NDX))
        self.is_train = is_train
        self.train_len = int(0.7 * len(self.X))
        self.test_len = len(self.X) - self.train_len
        self.device = opt.device

    def __getitem__(self, item):
        if self.is_train:
            self.return_x = self.X[item: item + self.T, :]
            self.return_y_history = self.y[item: item + self.T - 1]
            self.return_y_target = self.y[item + self.T]
        else:
            self.return_x = self.X[self.train_len + item: self.train_len + item + self.T, :]
            self.return_y_history = self.y[self.train_len + item: self.train_len + item + self.T - 1]
            self.return_y_target = self.y[self.train_len + item + self.T]
        return self.return_x, self.return_y_history,\
               self.return_y_target

    def __len__(self):
        if self.is_train:
            return self.train_len - self.T
        else:
            return self.test_len - self.T

    def preprocess_X_data(self, raw_data):
        scaler = StandardScaler().fit(raw_data)
        prerocess_data = scaler.transform(raw_data)
        return prerocess_data

    def preprocess_y_data(self, raw_data):
        return (raw_data - np.mean(raw_data)) / np.std(raw_data), np.mean(raw_data), np.std(raw_data)

def main(**kwargs):
    opt = Configuration()
    opt._setting(kwargs)

if __name__ == '__main__':
    fire.Fire(main)