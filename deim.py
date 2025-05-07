import os
import sys
import time
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from utility import compute_svd

class DEIM():
    def __init__(self, filename, basis, constraint=False):
        print('DEIM: greedily to find the best sensor locations')
        print('the number of sensors should not be larger than the number of the basis ...')
        self._filename    = filename
        self._basis       = basis
        self._n, self._nb = basis.shape
        self._constraint  = constraint
        self._indices     = None
        try:
            h5file = h5py.File(filename, 'r')
            print('Read interpolation points from '+filename+' ...')
            self._indices = h5file['indices'][()]
            if self._constraint:
                self._candidate_indices = h5file['candidate_indices'][()]
                self._local_indices = h5file['local_indices'][()]
            print('The DEIM has been fit. Use {:d} interpolation points ...'.format(self._nb))
        except (FileNotFoundError, OSError, AssertionError):
            print('The DEIM has not been fit. The indices should be selected by using fit first ...')

    def fit(self, candidate_indices=None):
        if self._indices is not None:
            print('DEIM indices have been read from: ' + self._filename)
            print('Skip fit ...')
            if self._constraint:
                return self._indices, self._local_indices, self._candidate_indices
            else:
                return self._indices

        print('Find and write interpolation points to ' + self._filename + " ...")
        self._indices = []
        # self._cond = []
        if self._constraint is False or candidate_indices is None:
            best_loc = np.argmax(abs(self._basis[:,0]))
            self._indices.append(best_loc)
            print(self._indices)
        elif self._constraint is True and candidate_indices is not None:
            # sorted candidate indices, don't need to be sorted!
            # candidate_indices = np.sort(candidate_indices)
            self._candidate_indices = candidate_indices
            self._local_indices = [] # local indices in the candidate indices
            print(candidate_indices)
            local_best_loc = np.argmax(abs(self._basis[candidate_indices,0]))
            self._local_indices.append(local_best_loc)
            self._indices.append(candidate_indices[local_best_loc])
            print(self._indices)
            print(self._local_indices)
        # self._cond.append(np.linalg.cond(self._basis[self._indices,:1]))
        for ib in range(1, self._nb):
            print('Round', ib)
            A   = self._basis[self._indices,:ib]
            rhs = self._basis[self._indices,ib]
            c   = np.linalg.solve(A,rhs)
            if candidate_indices is None:
                best_loc = np.argmax(abs(self._basis[:,ib]-np.sum(self._basis[:,:ib]*c,axis=1)))
                self._indices.append(best_loc)
                print(self._indices)
            else:
                local_best_loc = np.argmax(abs(self._basis[candidate_indices,ib] - np.sum(self._basis[candidate_indices,:ib]*c,axis=1)))
                self._local_indices.append(local_best_loc)
                self._indices.append(candidate_indices[local_best_loc])
                print(self._indices)
                print(self._local_indices)
            A   = self._basis[self._indices,:ib+1]
            # self._cond.append(np.linalg.cond(A))
            # print('cond_number={:.3e}'.format(self._cond[-1]))
        # self._cond = np.array(self._cond)
        h5file = h5py.File(self._filename, 'w')
        h5file['indices'] = self._indices
        if self._constraint:
            h5file['candidate_indices'] = self._candidate_indices
            h5file['local_indices'] = self._local_indices
        h5file.close()
        if self._constraint:
            return self._indices, self._local_indices, self._candidate_indices
        else:
            return self._indices

    def get_reconstruction_mat(self, ns):
        if ns > self._nb:
            print('DEIM: the number of possible sensors should not be larger than
                    the number of the basis ...')
        M = self._basis[:,:ns].dot(np.linalg.inv(self._basis[self._indices[:ns],:ns]))
        return M

    def predict(self, M, average, reduced_average, reduced_testing_data):
        print('DEIM: reconstruct the testing data with existing interpolation points ...')
        recon = M.dot(reduced_testing_data - reduced_average) + average
        # x = data_testing - average
        # A = self._basis[self._indices,:]
        # b = x[self._indices, :]
        # coeff = np.linalg.lstsq(A, b, rcond=None)[0]
        # recon = self._basis.dot(coeff) + average
        # print('DEIM sample error = {:.3e}'.format(np.sum((recon[self._indices,:] - data_testing[self._indices,:]) **2)))
        # diff  = np.sqrt(np.sum((recon - data_testing)**2, axis=0))
        # scale = np.sqrt(np.sum(data_testing**2, axis=0))
        # recon_err = np.sum(diff/scale)/n_test
        # recons.append(recon)
        # recon_errs.append(recon_err)
        return recon

