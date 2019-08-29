import os
import torch
import torch.nn as nn
from torch.utils import data
from utils import *
from model import *
from evaluation import *
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import tqdm
import fire


writer = SummaryWriter('runs/exp10')
opt = Configuration()
#
# SEED = 4321
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

class Trainer(object):
    def __init__(self, opt):
        self.model = Model(opt)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.criterion = nn.MSELoss()
        self.train_dataset = NasdaqDataset(opt, is_train=True)
        self.test_dataset = NasdaqDataset(opt, is_train=False)
        self.train_dataloader = data.DataLoader(self.train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                num_workers=opt.num_workers)
        self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=opt.batch_size, shuffle=False,
                                               num_workers=opt.num_workers)
        self.y_std = self.train_dataset.y_std
        self.y_mean = self.train_dataset.y_mean
        self.model.to(opt.device)

    def train(self):
        every_epoch_training_loss = []
        every_epoch_testing_loss = []
        for epoch in range(1, opt.epochs+1):
            mini_batch_loss = []
            pred_record = np.array([])
            target_record = np.array([])
            for x, y_history, y_target in self.train_dataloader:
                x, y_history, y_target = x.to(opt.device).float(), \
                                         y_history.to(opt.device).float(), y_target.to(opt.device).float()
                self.optimizer.zero_grad()
                y_pred = self.model(x, y_history).view(y_target.shape)
                y_pred_adjust = y_pred * self.y_std + self.y_mean
                y_target_adjust = y_target * self.y_std + self.y_mean
                y_pred_np = y_pred_adjust.detach().cpu().numpy()
                y_target_np = y_target_adjust.detach().cpu().numpy()
                pred_record = np.append(pred_record, y_pred_np)
                target_record = np.append(target_record, y_target_np)
                loss = self.criterion(y_pred_adjust, y_target_adjust)
                loss.backward()

                self.optimizer.step()
                mini_batch_loss.append(loss.item())
                eval_results = calculate_eval(y_pred_np, y_target_np)
                # print('epoch: %s |  loss: %.5f  |  RMSE: %.5f  | NRMSE: %.5f  |  MAPE: %.5f |  SMAPE: %.5f'
                #       % (epoch, loss.item(), eval_results[0], eval_results[1], eval_results[2], eval_results[3]))
            print('-' * 30)
            loss_in_this_epoch = np.mean(mini_batch_loss)
            every_epoch_training_loss.append(loss_in_this_epoch)
            eval_train = calculate_eval(pred_record, target_record)
            print('Training result: after %s epoch, training loss is %s' % (epoch, loss_in_this_epoch))
            print('RMSE: %.6f  |  NRMSE: %.6f  |  MAPE: %.6f  |  SMAPE: %.6f'
                  % (eval_train[0], eval_train[1], eval_train[2], eval_train[3]))
            print('-' * 30)
            loss_test_record, eval_test = self.test()
            self.model.train()
            loss_test = np.mean(loss_test_record)
            print(
                'Testing Result: epoch %s  |loss: %.5f  |  RMSE: %.5f  |  NRMSE: %.5f  |  MAPE: %.5f  |  SMAPE: %.5f' %
                (epoch, loss_test, eval_test[0], eval_test[1], eval_test[2], eval_test[3]))
            writer.add_scalar('Loss', loss_test, epoch)
            writer.add_scalar('RMSE', eval_test[0], epoch)
            writer.add_scalar('MAPE', eval_test[3], epoch)
            if len(every_epoch_testing_loss) == 0:
                torch.save(self.model, './checkpoints/best.pkl')
            elif loss_test < np.min(every_epoch_testing_loss):
                torch.save(self.model, './checkpoints/best.pkl')
            every_epoch_testing_loss.append(loss_test)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pred_record = np.array([])
        target_record = np.array([])
        loss_record = []
        for x, y_history, y_target in self.test_dataloader:
            x, y_history, y_target = x.to(opt.device).float(), \
                                     y_history.to(opt.device).float(), y_target.to(opt.device).float()
            y_pred = self.model(x, y_history).view(y_target.shape)
            y_pred_adjust = y_pred * self.y_std + self.y_mean
            y_target_adjust = y_target * self.y_std + self.y_mean
            y_pred_adjust_np = y_pred_adjust.detach().cpu().numpy()
            y_target_adjust_np = y_target_adjust.detach().cpu().numpy()
            pred_record = np.append(pred_record, y_pred_adjust_np)
            target_record = np.append(target_record, y_target_adjust_np)
            loss = self.criterion(y_pred_adjust, y_target_adjust)
            loss_record.append(loss.item())
        eval_results = calculate_eval(pred_record, target_record)
        return loss_record, eval_results


def main(**kwargs):
    opt._setting(kwargs)
    trainer = Trainer(opt)
    trainer.train()

if __name__ == '__main__':
    fire.Fire(main)