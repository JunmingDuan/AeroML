import sys, os
from os.path import dirname, join as pjoin
import scipy.io as scio
import argparse
import random
import time
import numpy as np
import h5py
import itertools
from matplotlib import pyplot as plt, colors
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from deim_nn_model import AeroCoeffModel
from deim_nn_data import AeroCoeffDataModule, GetDataset
from deim import DEIM

aref = 0.0225

def deim_nn_aero_coeff(v_ckpt, ns, data_dir):
    print('Read numerical and experimental data')
    numer_data = h5py.File('../../Data/Numerics/initial_mesh.hdf5', 'r')
    numer_xy = numer_data['xy_body'][()]
    sf_xy    = numer_data['sf_body'][()].T
    exper_data = h5py.File('../../Data/Experiment/processed_data/exper_sensor_location.hdf5', 'r')
    exper_xy = exper_data['sensor_location'][()]
    print(sf_xy.shape)

    basis_filename = '../../DEIM/numer_basis_nb=36.hdf5'
    print('Read basis from '+basis_filename+' ...')
    basis_file = h5py.File(basis_filename, 'r')
    basis      = basis_file['U'][()]
    average    = basis_file['average'][()]

    print('Read deim data')
    deim = DEIM('../../DEIM/numer_deim_constraint_nb=36.hdf5', basis, constraint=True)
    DEIM_indices = deim._indices[:ns]
    DEIM_candidate_indices = deim._candidate_indices
    M = deim.get_reconstruction_mat(ns)
    print('Build deim matrices')
    MF_s = sf_xy.dot(M)/aref
    MF_0 = (sf_xy.dot(average) - sf_xy.dot(M.dot(average[DEIM_indices]))).flatten()/aref

    # obtain correction aero_coeff from nn
    v, c = v_ckpt
    ckpt = v+c
    print("> Read in nn from " + ckpt)
    nn = AeroCoeffModel.load_from_checkpoint(ckpt, map_location=torch.device('cpu'))
    if ns != nn.args.n_in:
        print('The number of sensors is wrong!')
        exit()
    dm = AeroCoeffDataModule(nn.args)
    print("> Prepare data")
    dm.prepare_data()
    if nn.args.normalize:
        print("> Read normalization")
        dm.normalize()

    print("> Compute aero coefficients of test_set")
    nn.eval()
    dm.setup('test')

    instants = dm.test_h5['instants'][()]
    Cp_s_test= dm.test_h5['sensor'][()]
    Cl       = dm.test_h5['Cl'][()]
    Cd       = dm.test_h5['Cd'][()]
    Cl_body  = dm.test_h5['Cl_body'][()]
    Cd_body  = dm.test_h5['Cd_body'][()]
    aoa = dm.test_h5['aoa'][()]
    aoa_rad = aoa*np.pi/180

    DEIM_Cl_body = []
    DEIM_Cd_body = []
    nn_Cl_body = []
    nn_Cd_body = []
    n_test = len(aoa)
    DEIM_CPU_time = 0
    NN_CPU_time   = 0
    for i in range(n_test):
        # print("> Compute DEIM prediction of test_set")
        Cp_s = Cp_s_test[:,i]
        Cp_s *= 1 + 0.01*np.random.uniform(-noise, noise) # Gaussian noise
        t1 = time.time()
        DEIM_F = MF_s.dot(Cp_s) + MF_0 # DEIM_Cd_body, DEIM_Cl_body
        t2 = time.time()
        DEIM_CPU_time += t2-t1
        DEIM_Cl_body.append(DEIM_F[1])
        DEIM_Cd_body.append(DEIM_F[0])

        t3 = time.time()
        if dm.scale:
            Cp_s = (Cp_s - dm.x_mean) / dm.x_std
        with torch.no_grad():
            nn_aero_coeff_diff = nn(torch.from_numpy(Cp_s).float())
        nn_aero_coeff_diff    = nn_aero_coeff_diff.numpy()
        if dm.scale:
            nn_aero_coeff_diff    = nn_aero_coeff_diff*dm.y_std + dm.y_mean
        t4 = time.time()
        NN_CPU_time += t4-t3

        nn_Cl_body.append(DEIM_F[1] + nn_aero_coeff_diff[0])
        nn_Cd_body.append(DEIM_F[0] + nn_aero_coeff_diff[1])

    DEIM_CPU_time /= n_test
    NN_CPU_time /= n_test
    print('Averaged DEIM CPU time = {:.2e}'.format(DEIM_CPU_time))
    print('Averaged NN CPU time = {:.2e}'.format(NN_CPU_time))

    DEIM_Cl_body = np.array(DEIM_Cl_body)
    DEIM_Cd_body = np.array(DEIM_Cd_body)
    nn_Cl_body = np.array(nn_Cl_body)
    nn_Cd_body = np.array(nn_Cd_body)

    nn_Cl = nn_Cl_body*np.cos(aoa_rad) - nn_Cd_body*np.sin(aoa_rad) # to the true position
    nn_Cd = nn_Cl_body*np.sin(aoa_rad) + nn_Cd_body*np.cos(aoa_rad) # to the true position
    DEIM_Cl = DEIM_Cl_body*np.cos(aoa_rad) - DEIM_Cd_body*np.sin(aoa_rad) # to the true position
    DEIM_Cd = DEIM_Cl_body*np.sin(aoa_rad) + DEIM_Cd_body*np.cos(aoa_rad) # to the true position

    filename = os.path.join(data_dir, 'numer_deim_nn_ns_'+str(ns)+'.hdf5')
    print('> Hyper parameter for the NN is in '+os.path.join(v,'hparams.yaml'))
    print("> Save data to " + filename + ' for postprocessing ...')
    with open(os.path.join(v,'hparams.yaml'), 'r') as hp:
        print(hp.read())
    h5file = h5py.File(filename, 'w')
    h5file['numer_xy'] = numer_xy
    h5file['exper_xy'] = exper_xy
    h5file['DEIM_candidate_indices'] = DEIM_candidate_indices
    h5file['DEIM_indices'] = DEIM_indices
    h5file['instants'] = instants
    h5file['aoa'] = aoa
    h5file['Cl'] = Cl
    h5file['Cd'] = Cd
    h5file['Cl_body'] = Cl_body
    h5file['Cd_body'] = Cd_body
    h5file['DEIM_Cl'] = DEIM_Cl
    h5file['DEIM_Cd'] = DEIM_Cd
    h5file['DEIM_Cl_body'] = DEIM_Cl_body
    h5file['DEIM_Cd_body'] = DEIM_Cd_body
    h5file['nn_Cl'] = nn_Cl
    h5file['nn_Cd'] = nn_Cd
    h5file['nn_Cl_body'] = nn_Cl_body
    h5file['nn_Cd_body'] = nn_Cd_body
    h5file['DEIM_CPU_time'] = DEIM_CPU_time
    h5file['NN_CPU_time'] = NN_CPU_time
    h5file.close()

    # compute errors
    n_inst = len(Cl)
    DEIM_Cl_diff = abs(DEIM_Cl - Cl)
    DEIM_Cd_diff = abs(DEIM_Cd - Cd)
    DEIM_Cl_rel_diff = DEIM_Cl_diff/Cl
    DEIM_Cd_rel_diff = DEIM_Cd_diff/Cd
    DEIM_Cl_l2_err = np.sqrt(np.sum(DEIM_Cl_diff**2)/n_inst)
    DEIM_Cd_l2_err = np.sqrt(np.sum(DEIM_Cd_diff**2)/n_inst)
    DEIM_Cl_rel_l2_err = np.sqrt(np.sum(DEIM_Cl_rel_diff**2)/n_inst)
    DEIM_Cd_rel_l2_err = np.sqrt(np.sum(DEIM_Cd_rel_diff**2)/n_inst)

    nn_Cl_diff = abs(nn_Cl - Cl)
    nn_Cd_diff = abs(nn_Cd - Cd)
    nn_Cl_rel_diff = nn_Cl_diff/Cl
    nn_Cd_rel_diff = nn_Cd_diff/Cd
    nn_Cl_l2_err = np.sqrt(np.sum(nn_Cl_diff**2)/n_inst)
    nn_Cd_l2_err = np.sqrt(np.sum(nn_Cd_diff**2)/n_inst)
    nn_Cl_rel_l2_err = np.sqrt(np.sum(nn_Cl_rel_diff**2)/n_inst)
    nn_Cd_rel_l2_err = np.sqrt(np.sum(nn_Cd_rel_diff**2)/n_inst)

    ind = np.argmax(DEIM_Cl_diff)
    DEIM_Cl_max_err = DEIM_Cl_diff[ind]
    aoa_DEIM_Cl_max_err = aoa[ind]
    ind = np.argmax(DEIM_Cl_rel_diff)
    DEIM_Cl_max_rel_err = DEIM_Cl_rel_diff[ind]
    aoa_DEIM_Cl_max_rel_err = aoa[ind]

    ind = np.argmax(DEIM_Cd_diff)
    DEIM_Cd_max_err = DEIM_Cd_diff[ind]
    aoa_DEIM_Cd_max_err = aoa[ind]
    ind = np.argmax(DEIM_Cd_rel_diff)
    DEIM_Cd_max_rel_err = DEIM_Cd_rel_diff[ind]
    aoa_DEIM_Cd_max_rel_err = aoa[ind]

    ind = np.argmax(nn_Cl_diff)
    nn_Cl_max_err = nn_Cl_diff[ind]
    aoa_nn_Cl_max_err = aoa[ind]
    ind = np.argmax(nn_Cl_rel_diff)
    nn_Cl_max_rel_err = nn_Cl_rel_diff[ind]
    aoa_nn_Cl_max_rel_err = aoa[ind]

    ind = np.argmax(nn_Cd_diff)
    nn_Cd_max_err = nn_Cd_diff[ind]
    aoa_nn_Cd_max_err = aoa[ind]
    ind = np.argmax(nn_Cd_rel_diff)
    nn_Cd_max_rel_err = nn_Cd_rel_diff[ind]
    aoa_nn_Cd_max_rel_err = aoa[ind]

    sample = 389
    # plots
    fig, axs = plt.subplots(2,1, figsize=(12,8))
    axs[0].plot(aoa[:sample], Cl[:sample], '-k', label='ground truth Cl')
    axs[0].plot(aoa[:sample], DEIM_Cl[:sample], 'or', markerfacecolor='none', markersize=3, label='DEIM Cl')
    axs[0].plot(aoa[:sample], nn_Cl[:sample], 'sb', markerfacecolor='none', markersize=3, label='DEIM+NN Cl')
    axs[0].set_xlabel('aoa')
    axs[0].set_ylabel('Cl')
    axs[0].legend()
    title = 'Abs DEIM Cl l2 error = {:.3e}; Rel DEIM Cl l2 error = {:.3e}\n'\
            +'Abs nn Cl l2 error = {:.3e}; Rel nn Cl l2 error = {:.3e}\n'\
            +'DEIM max error = {:.3e} at aoa = {:.3e}; DEIM max Rel error = {:.3e} at aoa = {:.3e}\n'\
            +'nn max error = {:.3e} at aoa = {:.3e}; nn max Rel error = {:.3e} at aoa = {:.3e}'
    axs[0].set_title(title.format(DEIM_Cl_l2_err, DEIM_Cl_rel_l2_err,\
            nn_Cl_l2_err, nn_Cl_rel_l2_err,\
            DEIM_Cl_max_err, aoa_DEIM_Cl_max_err, DEIM_Cl_max_rel_err, aoa_DEIM_Cl_max_rel_err,\
            nn_Cl_max_err, aoa_nn_Cl_max_err, nn_Cl_max_rel_err, aoa_nn_Cl_max_rel_err))

    axs[1].plot(aoa[:sample], Cd[:sample], '-k', label='ground truth Cd')
    axs[1].plot(aoa[:sample], DEIM_Cd[:sample], 'or', markerfacecolor='none', markersize=3, label='DEIM Cd')
    axs[1].plot(aoa[:sample], nn_Cd[:sample], 'sb', markerfacecolor='none', markersize=3, label='DEIM+NN Cd')
    axs[1].set_xlabel('aoa')
    axs[1].set_ylabel('Cd')
    axs[1].legend()
    title = 'Abs DEIM Cd l2 error = {:.3e}; Rel DEIM Cd l2 error = {:.3e}\n'\
            +'Abs nn Cd l2 error = {:.3e}; Rel nn Cd l2 error = {:.3e}\n'\
            +'DEIM max error = {:.3e} at aoa = {:.3e}; DEIM max Rel error = {:.3e} at aoa = {:.3e}\n'\
            +'nn max error = {:.3e} at aoa = {:.3e}; nn max Rel error = {:.3e} at aoa = {:.3e}'
    axs[1].set_title(title.format(DEIM_Cd_l2_err, DEIM_Cd_rel_l2_err,\
            nn_Cd_l2_err, nn_Cd_rel_l2_err,\
            DEIM_Cd_max_err, aoa_DEIM_Cd_max_err, DEIM_Cd_max_rel_err, aoa_DEIM_Cd_max_rel_err,\
            nn_Cd_max_err, aoa_nn_Cd_max_err, nn_Cd_max_rel_err, aoa_nn_Cd_max_rel_err))

    fig.suptitle(str(ns)+' bases and '+str(ns)+' sensors')
    fig.tight_layout()
    plt.show()

    fig2, axs = plt.subplots(2,1, figsize=(12,8))
    axs[0].plot(instants, Cl, '-k', label='ground truth Cl')
    axs[0].plot(instants, DEIM_Cl, 'or', markerfacecolor='none', markersize=3, label='DEIM Cl')
    axs[0].plot(instants, nn_Cl, 'sb', markerfacecolor='none', markersize=3, label='DEIM+NN Cl')
    axs[0].set_xlabel('instants')
    axs[0].set_ylabel('Cl')
    axs[0].legend()
    # title = 'Abs DEIM Cl l2 error = {:.3e}; Rel DEIM Cl l2 error = {:.3e}\n'\
            # +'Abs nn Cl l2 error = {:.3e}; Rel nn Cl l2 error = {:.3e}\n'\
            # +'DEIM max error = {:.3e} at aoa = {:.3e}; DEIM max Rel error = {:.3e} at aoa = {:.3e}\n'\
            # +'nn max error = {:.3e} at aoa = {:.3e}; nn max Rel error = {:.3e} at aoa = {:.3e}'
    # axs[0].set_title(title.format(DEIM_Cl_l2_err, DEIM_Cl_rel_l2_err,\
            # nn_Cl_l2_err, nn_Cl_rel_l2_err,\
            # DEIM_Cl_max_err, aoa_DEIM_Cl_max_err, DEIM_Cl_max_rel_err, aoa_DEIM_Cl_max_rel_err,\
            # nn_Cl_max_err, aoa_nn_Cl_max_err, nn_Cl_max_rel_err, aoa_nn_Cl_max_rel_err))

    axs[1].plot(instants, Cd, '-k', label='ground truth Cd')
    axs[1].plot(instants, DEIM_Cd, 'or', markerfacecolor='none', markersize=3, label='DEIM Cd')
    axs[1].plot(instants, nn_Cd, 'sb', markerfacecolor='none', markersize=3, label='DEIM+NN Cd')
    axs[1].set_xlabel('instants')
    axs[1].set_ylabel('Cd')
    axs[1].legend()
    # title = 'Abs DEIM Cd l2 error = {:.3e}; Rel DEIM Cd l2 error = {:.3e}\n'\
            # +'Abs nn Cd l2 error = {:.3e}; Rel nn Cd l2 error = {:.3e}\n'\
            # +'DEIM max error = {:.3e} at aoa = {:.3e}; DEIM max Rel error = {:.3e} at aoa = {:.3e}\n'\
            # +'nn max error = {:.3e} at aoa = {:.3e}; nn max Rel error = {:.3e} at aoa = {:.3e}'
    # axs[1].set_title(title.format(DEIM_Cd_l2_err, DEIM_Cd_rel_l2_err,\
            # nn_Cd_l2_err, nn_Cd_rel_l2_err,\
            # DEIM_Cd_max_err, aoa_DEIM_Cd_max_err, DEIM_Cd_max_rel_err, aoa_DEIM_Cd_max_rel_err,\
            # nn_Cd_max_err, aoa_nn_Cd_max_err, nn_Cd_max_rel_err, aoa_nn_Cd_max_rel_err))

    fig2.suptitle(str(ns)+' bases and '+str(ns)+' sensors')
    fig2.tight_layout()
    plt.show()

if __name__ == '__main__':
    noise_list = [0, 1.5]
    ns_list = [5, 8, 10]
    # grid_result = np.loadtxt('basis'+str(ns)+'/grid_results.csv', delimiter=',', skiprows=1)
    # grid_result = grid_result[grid_result[:,4].argsort(), :]
    # print(grid_result)
    # ko

    ckpts = [["basis5/training_logs/lightning_logs/version_11", "/checkpoints/epoch=18-step=2584-val_loss=1.287397e-01.ckpt"], # 5
    ["basis8/training_logs/lightning_logs/version_18", "/checkpoints/epoch=133-step=18224-val_loss=2.319190e-02.ckpt"], # 8
    ["basis10/training_logs/lightning_logs/version_24", "/checkpoints/epoch=441-step=60112-val_loss=2.877531e-02.ckpt"]] # 10

    for noise in noise_list:
        data_dir = os.path.join('/home/jduan/bigdisk/AeroLoadsWithSensors/2DAirfoil/PostProcess/post_data', 'noise='+str(noise)+'%')
        for i in range(3):
            deim_nn_aero_coeff(ckpts[i], ns_list[i], data_dir)

