
import sys
import warnings
import numpy as np
import torch
from sklearn import decomposition

def init_dct(n, m):
    """
    Overcomplete DCT dictionary
    """
    oc_dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)  # demean
        oc_dictionary[:, k] = V / np.linalg.norm(V)  # Normalize

    idx = np.arange(0, n)
    idx = idx.reshape(n, 1, order="F")
    idx = idx.reshape(n, order="C")
    oc_dictionary = oc_dictionary[idx, :]
    oc_dictionary = torch.from_numpy(oc_dictionary).float()
    return oc_dictionary


def init_PCA(X, N):
    '''
    Compute the N EOFs.
    '''
    dict = 0.1 * np.ones((X.shape[1], N))
    pca = decomposition.PCA()
    pca.fit(X)
    dict[:, 0:X.shape[1]] = pca.components_
    idx = np.arange(0, N)
    idx = idx.reshape(N, 1, order="F")
    idx = idx.reshape(N, order="C")
    dict = torch.from_numpy(dict).float()
    return dict


def init_Halfsin(x, P):
    '''
    initialize a Dictionary, using data
    '''
    P = int(P)
    N = x.shape[1]
    dict = np.zeros((N, P))
    phase = 10
    p_2 = int(phase/2)
    z = np.linspace(p_2, N + p_2, num=P, endpoint=True, retstep=False).reshape(P, 1)
    axe = np.linspace(1, N + phase, num=N + phase, endpoint=True, retstep=False).reshape(N + phase, 1)
    A = (2 * np.random.random(P) - 1).reshape(P, 1)
    # Amplitude for all columns

    for i in range(0, phase, 1):
        y = A[i] * np.sin((np.pi / (phase*2)) * (axe - z[i] + phase))
        # y[0: int(z[i]) - 10] = 0
        y[(int(z[i]) + phase):N + phase] = 0
        y = y[(p_2-1):N+(p_2-1)]
        dict[:, i] = np.squeeze(y)

    for i in range(phase, 100, 1):
        y = A[i] * np.sin((np.pi / (phase*2)) * (axe - z[i] + phase))
        y[0: int(z[i]) - phase] = 0
        y[(int(z[i]) + phase):N + phase] = 0
        y = y[(p_2-1):N+(p_2-1)]
        dict[:, i] = np.squeeze(y)

    for i in range(100, P, 1):
        y = A[i] * np.sin((np.pi / (phase*2)) * (axe - z[i] + phase))
        y[0: int(z[i]) - phase] = 0
        y = y[(p_2-1):N+(p_2-1)]
        dict[:, i] = np.squeeze(y)

    dict = torch.from_numpy(dict).float()
    return dict


def sech(x):
    e = np.e
    y = 2 / (e ** x + e ** (-x))
    return y


def generate_disturbance(depth, numsample, duration, t0, d0, deltad, eta0):
    """
    :param depth: No. of depth points
    :param numsample: No. of SSP samples
    :param duration: Duration of the disturbance
    :param t0: Time of the peak of the disturbance
    :param d0: Depth of the peak of the disturbance
    :param deltad: Depth range
    :param eta0: Amplitude
    :return:
    """
    Disturbance = np.zeros((depth, numsample))
    x = np.linspace(0, depth, depth, dtype=int)
    x.shape = (1, depth)
    depth_2 = int(depth/2)
    A = eta0 * (sech((x - depth_2) / duration * 4)) ** 2
    Amp = A[:, depth_2 - duration: depth_2 + duration + 1]
    x = np.linspace(0, depth, depth, dtype=int)
    x.shape = (1, depth)
    y = np.sin((np.pi / deltad) * (x + 1) + (np.pi / 2 - np.pi * d0 / deltad))
    y[:, 0: d0 - np.int(deltad / 2) - 1] = 0
    y[:, d0 + np.int(deltad / 2): depth] = 0
    temp = 0
    for i in range(t0 - duration - 1, t0 + duration):
        Disturbance[:, i] = Amp[:, temp] * y
        temp = temp + 1

    return Disturbance


def MixGausNoise(data, sigma1, sigma2, rate):
    '''

    :param data: input data
    :param sigma1: sigma of Gaussian noise model 1
    :param sigma2: sigma of Gaussian noise model 2
    :param rate: rate of Gaussian noise model 2
    :return: a matrix of mixed Gaussian noise of the same size of the input data
    '''
    if rate > 1 or rate < 0:
        warnings.warn("The input does not meet the requirements! rate needs to be between 0 and 1.", UserWarning)
        sys.exit(0)

    MixedGaussianNoise = np.zeros([data.shape[0], data.shape[1]])
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            flag = np.random.rand(1)
            if flag > rate:
                MixedGaussianNoise[i, j] = np.random.normal(0, sigma1, 1)
            else:
                MixedGaussianNoise[i, j] = np.random.normal(0, sigma2, 1)

    return MixedGaussianNoise


class SSP_Net(torch.nn.Module):
    def __init__(
            self,
            D_in,
            H_1,
            H_2,
            H_3,
            D_out_lam,
            T,
            Dict_init,
            c_init,
            device,
    ):
        super(SSP_Net, self).__init__()

        self.T = T

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)

        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)  #

        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)


    def soft_thresh(self, x, l):

        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)
        # sgn(x)*max(|x|-l,0)

    def forward(self, x_input, N):
        lin = self.linear1(x_input).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        lam = lam + 2
        l = lam / self.c
        y = torch.matmul(x_input, self.Dict)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        # S = I-1/c D

        S = S.t()

        alpha = self.soft_thresh(y, l)
        for t in range(self.T):
            alpha = self.soft_thresh(torch.matmul(alpha, S) + (1 / self.c) * y, l)
            # ISTA

        x_pred = torch.matmul(alpha, self.Dict.t())
        alpha_sparse = alpha
        res = x_pred
        return res, alpha_sparse, lam
