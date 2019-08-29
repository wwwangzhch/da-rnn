import os
import torch
import torch.nn as nn
from torch.utils import data
from utils import *
from model import *
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import tqdm
import fire


writer = SummaryWriter('runs/exp')
opt = Configuration()

# SEED = 4321
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

def cal_RMSE(pred, target):
    return np.sqrt(np.mean((np.array(pred)-np.array(target))**2))


def cal_RRSE(pred, target):
    pred, target = np.array(pred), np.array(target)
    up = np.sum((target-pred)**2)
    down = np.sum((target-np.mean(target))**2)
    return np. sqrt(up/down)


def get_y_back(y, y_mean, y_std):
    return y * y_std + y_mean


class DA_RNN(object):
    def __init__(self, opt):
        self.T = opt.T
        self.train_dataset = NasdaqDataset(opt)
        self.test_dataset = NasdaqDataset(opt, is_train=False)
        self.model = Model(opt).to(opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # self.encoder = Encoder(opt).to(opt.device)
        # self.decoder = Decoder(opt).to(opt.device)
        # self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # self.opt_decoder = torch.optim.Adam(self.decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.train_dataloader = data.DataLoader(self.train_dataset, batch_size=opt.batch_size, shuffle=False,
                                                num_workers=opt.num_workers)
        self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=16*opt.batch_size,
                                               num_workers=opt.num_workers)
        self.loss_func = nn.MSELoss()
        self.y_std = self.train_dataset.y_std
        self.y_mean = self.train_dataset.y_mean

    def train(self):
        print('-'*10, '>'*3, 'Training Start', '<'*3, '-'*10)
        training_start = time.time()
        every_epoch_training_loss = []
        every_epoch_testing_loss = []
        for i in range(1, opt.epochs+1):
            print('-'*30)
            time.sleep(0.01)
            start = time.time()
            epoch_loss = []
            pred_record = np.array([])
            target_record = np.array([])
            for x, y_history, y_target in tqdm.tqdm(self.train_dataloader):
                x, y_history, y_target = x.to(opt.device).float(), y_history.to(opt.device).float(), \
                                         y_target.to(opt.device).float()
                # self.opt_encoder.zero_grad()
                # self.opt_decoder.zero_grad()
                self.optimizer.zero_grad()
                y_pred = self.model(x, y_history).view(y_target.shape)
                # input_weighted, input_encoded = self.encoder(x)
                # y_pred = self.decoder(input_encoded, y_history).view(y_target.shape)
                y_pred_adjust = y_pred * self.y_std + self.y_mean
                y_target_adjust = y_target * self.y_std + self.y_mean
                pred_record = np.append(pred_record, y_pred_adjust.detach().cpu().numpy())
                target_record = np.append(target_record, y_target_adjust.detach().cpu().numpy())
                loss = self.loss_func(y_pred_adjust, y_target_adjust)
                loss.backward()

                # self.opt_encoder.step()
                # self.opt_decoder.step()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            end = time.time()
            print('Training result:'
                  'after %s epoch, training loss is %s, this epoch takes %.2fs, the total epoch takes %.2fs'
                  % (i, np.mean(epoch_loss), end-start, end-training_start))
            if len(every_epoch_training_loss) == 0 or np.mean(epoch_loss) < np.min(every_epoch_training_loss):
                torch.save(self.model, 'model.pkl')
            every_epoch_training_loss.append(np.mean(epoch_loss))
            test_loss = self.test(i)
            train_loss = np.mean(epoch_loss)
            self.vis(i, train_loss, test_loss)
        l1 = plt.plot(target_record, label='True')
        # self.decoder = Decoder(opt).to(opt.device)
        # self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # self.opt_decoder = torch.optim.Adam(self.decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.train_dataloader = data.DataLoader(self.train_dataset, batch_size=opt.batch_size, shuffle=False,
                                                num_workers=opt.num_workers)
        self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=4*opt.batch_size,
                                               num_workers=opt.num_workers)
        self.loss_func = nn.MSELoss()
        l2 = plt.plot(pred_record, label='Predict')
        plt.legend()
        plt.show()

    @torch.no_grad()
    def test(self, epoch):
        # self.encoder.eval()
        # self.decoder.eval()
        self.model.eval()
        test_loss = []
        pred_record = np.array([])
        target_record = np.array([])
        for x, y_history, y_target in self.test_dataloader:
            x, y_history, y_target = x.to(opt.device).float(), y_history.to(opt.device).float(), \
                                     y_target.to(opt.device).float()
            # input_weighted, input_encoded = self.encoder(x)
            # y_pred = self.decoder(input_encoded, y_history).view(y_target.shape)
            y_pred = self.model(x, y_history).view(y_target.shape)
            y_pred_adjust = y_pred * self.y_std + self.y_mean
            y_target_adjust = y_target * self.y_std + self.y_mean
            pred_record = np.append(pred_record, y_pred_adjust.cpu())
            target_record = np.append(target_record, y_target_adjust.cpu())
            loss = self.loss_func(y_pred_adjust, y_target_adjust)
            test_loss.append(loss.item())
        # self.decoder.train()
        # self.encoder.train()
        self.model.train()
        RMSE = cal_RMSE(pred_record, target_record)
        RRSE = cal_RRSE(pred_record, target_record)
        print('Testing result: the test loss is %.5f, the RMSE is %.5f, the RRSE is %.5f' % (np.mean(np.array(test_loss)), RMSE, RRSE))
        writer.add_scalar('RMSE', RMSE, epoch)
        writer.add_scalar('RRSE', RRSE, epoch)
        return np.mean(np.array(test_loss))

    def vis(self, epoch, train_loss, test_loss):
        writer.add_scalars('loss', {
            'train': train_loss,
            'test': test_loss
        }, epoch)


def main(**kwargs):
    opt._setting(kwargs)
    darnn = DA_RNN(opt)
    darnn.train()

if __name__ == '__main__':
    fire.Fire(main)
    # darnn = DA_RNN()
    # darnn.train()