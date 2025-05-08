import numpy as np

def compute_svd(X, svd_rank = 0):
    '''
    svd_rank = positive integer for truncation,
    svd_rank = 0 for optimal
    0 < svd_rank < 1 for biggest singular values that are needed to reach the 'energy'
    svd_rank = -1 for no truncation
    '''
    U, s, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T

    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if svd_rank == 0:
        beta = np.divide(*sorted(X.shape))
        tau  = np.median(s) * omega(beta)
        rank = np.sum(s > tau)
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(s**2 / (s**2).sum())
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = X.shape[1]

    # import matplotlib.pyplot as plt
    # plt.semilogy(s/s.max())
    # plt.show()
    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]

    return U, s, V

'''
class Normalization():
    def __init__(self, mode='normal', isone_std=False):
        self.mode = mode
        self.isone_std = isone_std

    def compute_mapping(self, data, file_name):
        if self.mode == 'normal':
            data_mean = np.mean(data, 1) # mean of different snapshots
            data_std  = np.std(data, 1)
            # data_std[np.where(abs(data_std) < 1e-15)[0]] += 1 # avoid zero std
            data_std  = np.sqrt(data_std**2 + 1e-12) # avoid zero std
            data_mean = data_mean[:, np.newaxis]
            data_std  = data_std[:, np.newaxis]
            if self.isone_std == True:
                data_std  = np.ones(data_std.shape)
            self.mapping = (data_mean, data_std)
        elif self.mode == 'minmax':
            print('Not finished')
            tbd
        else:
            print('No such mode for', self.mode)

        np.save(file_name, self.mapping)
        print('> Writing mapping function to', file_name, '...')
        return self.mapping

    def read_mapping(self, file_name):
        if self.mode == 'normal':
            self.mapping = np.load(file_name, allow_pickle=True)
        elif self.mode == 'minmax':
            print('Not finished')
            tbd

    def trs(self, data, direction='forward'):
        if data.ndim == 1:
            data = data[:,None]
        if self.mode == 'normal':
            data_mean, data_std = self.mapping
            if direction == 'forward':
                if self.isone_std == True:
                    data = (data - data_mean)
                else:
                    data = (data - data_mean)/data_std
            elif direction == 'backward':
                if self.isone_std == True:
                    data = data + data_mean
                else:
                    data = data*data_std + data_mean
            else:
                print('Wrong choice!')
                mkm
        return data

    def map_minmax(data):
        lb = data.min(axis=1)
        h_lb = lb[:nx].min()
        v_lb = lb[nx:].min()
        lb = np.vstack((h_lb*np.ones((nx,1)), v_lb*np.ones((nx,1))))

        ub = data.max(axis=1)
        h_ub = ub[:nx].max()
        v_ub = ub[nx:].max()
        ub = np.vstack((h_ub*np.ones((nx,1)), v_ub*np.ones((nx,1))))

        data = (data-lb)/(ub-lb)*2-1
        mapping = (lb, ub,)
        return data, mapping
    def rec_map_minmax(data,mapping):
        lb, ub = mapping
        data = (data+1)/2*(ub-lb) +lb
        return data
    def map_max(data):
        ub = data.max(axis=1)
        # h_ub = ub[:nx].max()
        # v_ub = ub[nx:].max()
        # ub = np.vstack((h_ub*np.ones((nx,1)), v_ub*np.ones((nx,1))))
        # ub = h_ub*np.ones((nx,1))
        ub = ub[:, np.newaxis]

        data = data/ub
        mapping = (ub,)
        return data, mapping
    def rec_map_max(data,mapping):
        ub = mapping
        data = data*ub
        return data
'''

class Normalization():
    def __init__(self, mode='normal', isone_std=False):
        self.mode = mode
        self.isone_std = isone_std

    def compute_mapping(self, data, file_name):
        # data is of (nt*np)-by-(nv*ny*nx)
        shape = data.shape
        r_data = data.reshape((shape[0], -1))
        if self.mode == 'normal':
            data_mean = np.mean(r_data, 0) # mean of different snapshots
            data_std  = np.std(r_data, 0)
            # data_std[np.where(abs(data_std) < 1e-15)[0]] += 1 # avoid zero std
            data_std  = np.sqrt(data_std**2 + 1e-12) # avoid zero std
            data_mean = data_mean.reshape(shape[1:])
            data_std  = data_std.reshape(shape[1:])
            if self.isone_std == True:
                data_std  = np.ones(data_std.shape)
            self.mapping = (data_mean, data_std)
        elif self.mode == 'minmax':
            print('Not finished')
            tbd
        else:
            print('No such mode for', self.mode)

        np.save(file_name, self.mapping)
        print('> Writing mapping function to', file_name, '...')
        return

    def read_mapping(self, file_name):
        if self.mode == 'normal':
            self.mapping = np.load(file_name, allow_pickle=True)
        elif self.mode == 'minmax':
            print('Not finished')
            tbd

    def trs(self, data, direction='forward'):
        if self.mode == 'normal':
            data_mean, data_std = self.mapping
            if direction == 'forward':
                if self.isone_std == True:
                    data = (data - data_mean)
                else:
                    data = (data - data_mean)/data_std
            elif direction == 'backward':
                if self.isone_std == True:
                    data = data + data_mean
                else:
                    print(data.shape)
                    print(data_mean.shape)
                    print(data_std.shape)
                    data = data*data_std + data_mean
            else:
                print('Wrong choice!')
                mkm
        return data

