from os.path import dirname, join as pjoin
import itertools
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from matplotlib import pyplot as plt

class GetDataset(Dataset):
    def __init__(self, stage, h5file):
        self.stage     = stage
        # self.para_list = para_list # pitch omega
        # self.inst_list = inst_list
        # self.root_dir  = root_dir
        self.h5file    = h5file
        self.scale     = []
        # self.n_para    = len(para_list)
        # self.instants  = self.h5file[0].attrs['instants'].squeeze()
        # self.n_inst    = []
        # for i in range(self.n_inst):
            # self.n_inst[i] = len(self.inst_list[i])
        # self.cumsum    = np.cumsum(self.n_list)
        # self.n_sample  = self.cumsum[-1]
        self.sensor       = h5file['sensor'][()]
        self.Cl_body_diff = h5file['Cl_body'][()] - h5file['DEIM_Cl_body'][()]
        self.Cd_body_diff = h5file['Cd_body'][()] - h5file['DEIM_Cd_body'][()]
        self.n_sample  = self.sensor.shape[1]
        print('sensor data shape:', self.sensor.shape)

    def __len__(self):
        return self.n_sample

    def set_scale(self, scale):
        self.scale = scale
        self.x_mean, self.x_std, self.y_mean, self.y_std = scale

    def __getitem__(self, idx):
        # idx_para = np.searchsorted(self.cumsum, idx)
        # idx_inst = int(idx - self.cumsum[idx_para])
        # print(idx, idx_para, idx_inst)
        # kok

        # time  = self.inst_list[None,idx_inst]
        # omega = self.inst_list[None,idx_inst]
        # amp   = self.inst_list[None,idx_inst]
        data = self.sensor[:,idx]
        # aoa  = 20 + np.sin(2*np.pi*omega*time)
        # aoa  = torch.from_numpy().float().squeeze()
        # para = torch.from_numpy(self.para_list[None,idx_para]).float().squeeze()
        data = torch.from_numpy(data).float()
        aeroCoeff = torch.from_numpy(np.array([self.Cl_body_diff[idx], self.Cd_body_diff[idx]])).float()

        if self.scale:
            data = (data - self.x_mean) / self.x_std
            aeroCoeff = (aeroCoeff - self.y_mean) / self.y_std

        return data, aeroCoeff

class AeroCoeffDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scale = []
        self.rootdir_snapshots = '/home/jduan/bigdisk/AeroLoadsWithSensors/2DAirfoil/Data'
        self.is_setup = False

    def prepare_data(self):
        pass

    def normalize(self):
        scaling_path = '/data/jduan/AeroLoadsWithSensors/2DAirfoil/NN/numer_DEIM_NN/basis'+str(self.args.n_in)+'/scaling.hdf5'
        try:
            with h5py.File(scaling_path, 'r') as f:
                print('Reading scaling for the training dataset from', scaling_path)
                self.x_mean = f['x_mean'][()]
                self.x_std  = f['x_std'][()]
                self.y_mean = f['y_mean'][()]
                self.y_std  = f['y_std'][()]
        except FileNotFoundError:
            print('Perform scaling for the training dataset first ...')
            self.setup('fit')
            dl = DataLoader(self.train_set, batch_size=len(self.train_set), shuffle=False)
            with torch.no_grad():
                for batch in dl:
                    x, y = batch
                    self.x_mean = torch.mean(x, axis=0)
                    self.x_std  = torch.std(x, axis=0)
                    self.x_std  = torch.sqrt(self.x_std**2 + 1e-12) # avoid zero std
                    self.y_mean = torch.mean(y, axis=0)
                    self.y_std  = torch.std(y, axis=0)
                    self.y_std  = torch.sqrt(self.y_std**2 + 1e-12) # avoid zero std
            with h5py.File(scaling_path, 'w') as f:
                print('Writing scaling for the training dataset to', scaling_path)
                f['x_mean'] = self.x_mean
                f['x_std']  = self.x_std
                f['y_mean'] = self.y_mean
                f['y_std']  = self.y_std
                # index = np.where(abs(data_std) < 1e-6)
                # data_std[index] += 1 # avoid zero std
        self.scale = [self.x_mean, self.x_std, self.y_mean, self.y_std]

    def setup(self, stage=None):
        if self.is_setup == False:
            # initialize the parameters and instants of the dataset
            # self.train_para = np.loadtxt(self.rootdir_snapshots+'/training/train_para.csv', delimiter=',')
            # self.val_para = np.loadtxt(self.rootdir_snapshots+'/validation/val_para.csv', delimiter=',')
            # self.test_para = np.loadtxt(self.rootdir_snapshots+'/testing/test_para.csv', delimiter=',')

            # self.train_inst = np.loadtxt(self.rootdir_snapshots+'/training/train_inst.csv', delimiter=',')
            # self.val_inst = np.loadtxt(self.rootdir_snapshots+'/validation/val_inst.csv', delimiter=',')
            # self.test_inst = np.loadtxt(self.rootdir_snapshots+'/testing/test_inst.csv', delimiter=',')

            self.train_h5 = h5py.File(self.rootdir_snapshots+'/training/numer_sensorPressureAeroCoeff_basis'+str(self.args.n_in)+'.hdf5')
            self.val_h5   = h5py.File(self.rootdir_snapshots+'/validation/numer_sensorPressureAeroCoeff_basis'+str(self.args.n_in)+'.hdf5')
            self.test_h5  = h5py.File(self.rootdir_snapshots+'/testing/numer_sensorPressureAeroCoeff_basis'+str(self.args.n_in)+'.hdf5')
            self.is_setup = True

        if stage == "fit" or stage is None:
            self.train_set = GetDataset('train', self.train_h5)
            self.val_set   = GetDataset('val', self.val_h5)
            if self.args.normalize and self.scale:
                self.train_set.set_scale(self.scale)
                self.val_set.set_scale(self.scale)

        if stage == "test" or stage is None:
            self.test_set  = GetDataset('test', self.test_h5)
            if self.args.normalize and self.scale:
                self.test_set.set_scale(self.scale)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=len(self.val_set), shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)

