import sys
from os.path import dirname, join as pjoin
from collections import OrderedDict
import numpy as np
import random
import time
from matplotlib import pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, torch.utils.data as Data
import pytorch_lightning as pl

random.seed(1234)

class AeroCoeffModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.mse_mean = nn.MSELoss(reduction = 'mean')
        self.args = args
        self.act  = nn.ReLU()
        net_dict = OrderedDict()
        self.n_layers = len(args.n_fc)-1
        for i in range(self.n_layers-1):
            net_dict['fc'+str(i)] = nn.Linear(args.n_fc[i], args.n_fc[i+1])
            net_dict['act'+str(i)] = self.act
        net_dict['fc'+str(self.n_layers-1)] = nn.Linear(args.n_fc[-2], args.n_fc[-1])
        self.fc = nn.Sequential(net_dict)
        self.ii = 0
        print(self.device)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.mse_mean(y, z)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.mse_mean(y, z)
        self.log("val_loss", loss, prog_bar=True)

        self.fig = plt.figure(1, figsize=(12, 8), facecolor='white')
        self.axs = self.fig.add_axes([0,0,1,1])

        self.axs.cla()

        Cl    = y.cpu().numpy()[:,0]
        Cl_nn = z.cpu().numpy()[:,0]
        Cd    = y.cpu().numpy()[:,1]
        Cd_nn = z.cpu().numpy()[:,1]
        self.axs.plot(Cl, '-r', label='Cl')
        self.axs.plot(Cl_nn, 'og', markerfacecolor='none', label='Cl_nn')
        self.axs.plot(Cd, '-b', label='Cd')
        self.axs.plot(Cd_nn, 'om', markerfacecolor='none', label='Cd_nn')
        self.fig.legend()
        # self.ax_valid1.axis('off')

        tensorboard = self.logger.experiment
        tensorboard.add_figure("val_fig", self.fig, global_step=self.ii, close=True)
        # self.ii += 1
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.mse_mean(y, z)
        self.log("test_loss", loss, prog_bar=True)

        self.fig = plt.figure(1, figsize=(12, 8), facecolor='white')
        self.axs = self.fig.add_axes([0,0,1,1])

        self.axs.cla()

        Cl    = y.cpu().numpy()[:,0]
        Cl_nn = z.cpu().numpy()[:,0]
        Cd    = y.cpu().numpy()[:,1]
        Cd_nn = z.cpu().numpy()[:,1]
        self.axs.plot(Cl, '-r', label='Cl')
        self.axs.plot(Cl_nn, 'og', markerfacecolor='none', label='Cl_nn')
        self.axs.plot(Cd, '-b', label='Cd')
        self.axs.plot(Cd_nn, 'om', markerfacecolor='none', label='Cd_nn')
        self.fig.legend()
        # self.ax_valid1.axis('off')

        tensorboard = self.logger.experiment
        tensorboard.add_figure("test_fig", self.fig, global_step=0, close=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.sch_step, gamma=self.args.sch_gamma, verbose=True)
        # return {"optimizer:": self.optimizer, "lr_scheduler": self.scheduler}
        return [self.optimizer], [self.scheduler]
        # return self.optimizer

