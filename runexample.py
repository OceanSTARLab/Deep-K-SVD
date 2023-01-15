
import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader
import time
import Deep_KSVD_For_SSP
from scipy import linalg
import scipy.io as sio

datatemp = sio.loadmat('data/ssp_mat.mat')
dataset = np.transpose(datatemp['ssp_mat'])
datatmp = sio.loadmat('data/ssp_hycom.mat')
ssp_hycom = np.transpose(datatmp['ssp_hycom'])

# train set
train_origin = dataset[0:9000, :]
train_mean = np.mean(train_origin, axis=0, keepdims=True)
train_demean = train_origin - train_mean

# test set 1 (SSP with noise)
test1_ori = dataset[9000:10000, :]
test1_mean = np.mean(test1_ori, axis=0, keepdims=True)
test1_demean = test1_ori - test1_mean

# test set 2 (Hycom)
test2_ori = ssp_hycom
test2_mean = np.mean(test2_ori, axis=0, keepdims=True)
test2_demean = test2_ori - test2_mean  #

# test set 3 (with disturbances)
test3_o = dataset[10000:10200, :]
##
depth = 60
numsample = 200
duration = 4
t0 = 70
d0 = 25
deltad = 19
eta0 = 12
# parameters of the disturbance
##

test3_demean = test3_o - np.mean(test3_o, axis=0, keepdims=True)
test3_o_p = test3_o
y5 = Deep_KSVD_For_SSP.generate_disturbance(depth, numsample, duration, t0, d0, deltad, eta0)
test3_o_p = test3_o_p + np.transpose(y5)
test3_demean_p = test3_o_p - np.mean(test3_o_p, axis=0, keepdims=True)

sigma = 0.9445
# Standard dev of noise

train_idx = np.linspace(0, len(train_demean) - 1, num=len(train_demean), dtype=int)
test1_idx = np.linspace(0, len(test1_demean) - 1, num=len(test1_demean), dtype=int)
test2_idx = np.linspace(0, len(test2_demean) - 1, num=len(test2_demean), dtype=int)
test3_idx = np.linspace(0, len(test3_demean) - 1, num=len(test3_demean), dtype=int)
test3_p_idx = np.linspace(0, len(test3_demean_p) - 1, num=len(test3_demean_p), dtype=int)
data_train = []
data_test1 = []
data_test2 = []
data_test3 = []
data_test3_p = []

for i in train_idx:
    data_train.append(train_demean[i])
for i in test1_idx:
    data_test1.append(test1_demean[i])
for i in test2_idx:
    data_test2.append(test2_demean[i])
for i in test3_idx:
    data_test3.append(test3_demean[i])
for i in test3_p_idx:
    data_test3_p.append(test3_demean_p[i])

batch_size = len(data_train)
dataloader_test1 = DataLoader(
    data_test1, batch_size=1, shuffle=False, num_workers=0
)
dataloader_train = DataLoader(
    data_train, batch_size=batch_size, shuffle=False, num_workers=0
)
dataloader_test2 = DataLoader(
    data_test2, batch_size=1, shuffle=False, num_workers=0
)
dataloader_test3 = DataLoader(
    data_test3, batch_size=1, shuffle=False, num_workers=0
)
dataloader_test3_p = DataLoader(
    data_test3_p, batch_size=1, shuffle=False, num_workers=0
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###################
N = 120
input_size = len(data_test1[1])
Dict_init = Deep_KSVD_For_SSP.init_Halfsin(np.array(data_test1), N)
dictarray = Dict_init.numpy()
Dict_init = Dict_init.to(device)
c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)


D_in, H_1, H_2, H_3, D_out_lam, T = \
    input_size, input_size * 2, input_size * 1, input_size, 1, 10

model = Deep_KSVD_For_SSP.SSP_Net(
    D_in,
    H_1,
    H_2,
    H_3,
    D_out_lam,
    T,
    Dict_init,
    c_init,
    device,
)
# Initialization of the model, the model is then overwritten.
###################

# model.load_state_dict(torch.load("model_noise_version.pth", map_location="cpu"))
model.load_state_dict(torch.load("model_disturbance_version.pth", map_location="cpu"))
model.to(device)
# model.T = 0

# Overwrite the model. For different proposes, load different models.


# loss function
criterion = torch.nn.MSELoss(reduction="mean")

N = 120
test_alphas = np.zeros((len(data_test1), int(N)))
test2_alphas = np.zeros((len(data_test2), int(N)))
test3_alphas = np.zeros((len(data_test3), int(N)))
all_test3_inputs = np.zeros((len(data_test3), 60))
all_test_outputs2 = np.zeros((len(data_test2), 60))
all_test_outputs3 = np.zeros((len(data_test3), 60))
all_test1_outputs = np.zeros((len(data_test1), 60))
test3_lam = np.zeros((len(data_test3), 1))
test1_lam = np.zeros((len(data_test1), 1))
i = 0
j = 0
k = 0

with torch.no_grad():
    test_loss = 0
    test_sploss = 0
    test2_loss = 0
    test2_sploss = 0
    test3_loss = 0
    test3_sploss = 0

    start = time.time()

    for test_inputs in dataloader_test1:
        # test 1
        test_inputs = (
            test_inputs.to(device),
        )
        test_inputs = test_inputs[0].to(torch.float32)
        test_inputs_noise = test_inputs
        test_inputs_noise = test_inputs_noise + sigma * np.random.randn(test_inputs.shape[0],
                                                                  test_inputs.shape[1])
        # test_inputs_noise = test_inputs_noise + Deep_KSVD_For_SSP.MixGausNoise(test_inputs, 0.2987, 5.3111, 0.01)
        # Mix Gaussian noise

        test_inputs_noise[0].to(torch.float32)
        test_inputs_noise = (torch.reshape(test_inputs_noise, (1, -1))).float()
        test_outputs, test_alpha, one_test1_lam = model(test_inputs_noise, N)
        test_alphas[i, :] = test_alpha
        loss = criterion(test_outputs, test_inputs)
        test_loss += loss.item()
        all_test1_outputs[i, :] = test_outputs
        test1_lam[i, :] = one_test1_lam
        i += 1

    end = time.time()
    time_curr = end - start
    print("test time:" + " " + str(time_curr) + "\n")

    for test2_inputs in dataloader_test2:
        # test 2 from Hycom
        test2_inputs = (
            test2_inputs.to(device),
        )
        test2_inputs = test2_inputs[0].to(torch.float32)
        test2_inputs = test2_inputs + sigma * np.random.randn(test2_inputs.shape[0],
                                                              test2_inputs.shape[1])
        test2_inputs = (torch.reshape(test2_inputs, (1, -1))).float()
        test2_inputs = test2_inputs.to(torch.float32)
        test2_outputs, test2_alpha, one_test2_lam = model(test2_inputs, N)

        test_input_ori = (torch.reshape(torch.tensor(data_test2[j]), (1, -1))).float()
        loss2 = criterion(test2_outputs, test_input_ori)
        test2_loss += loss2.item()
        test2_alphas[j, :] = test2_alpha
        all_test_outputs2[j, :] = test2_outputs
        j += 1

    for test3_inputs in dataloader_test3_p:  #
        # test 3
        test3_inputs = (
            test3_inputs.to(device),
        )
        test3_inputs = test3_inputs[0].to(torch.float32)
        test3_inputs = test3_inputs + sigma * np.random.randn(test3_inputs.shape[0],
                                                              test3_inputs.shape[1])
        test3_inputs = (torch.reshape(test3_inputs, (1, -1))).float()
        test3_inputs = test3_inputs.to(torch.float32)
        test3_outputs, test3_alpha, one_test3_lam = model(test3_inputs, N)
        test_input_ori = (torch.reshape(torch.tensor(data_test3[k]), (1, -1))).float()
        loss3 = criterion(test3_outputs, test_input_ori)
        test3_loss += loss3.item()
        test3_alphas[k, :] = test3_alpha
        all_test_outputs3[k, :] = test3_outputs
        all_test3_inputs[k, :] = test3_inputs
        test3_lam[k, :] = one_test3_lam
        k += 1

    test_loss = test_loss / len(dataloader_test1)
    test2_loss = test2_loss / len(dataloader_test2)
    test3_loss = test3_loss / len(dataloader_test3_p)

    RMSE_test_loss = np.sqrt(test_loss)
    RMSE_test2_loss = np.sqrt(test2_loss)
    RMSE_test3_loss = np.sqrt(test3_loss)

    print("\n" + "RMSE of test set 1(no dis):" + " " + str(RMSE_test_loss))
    print("RMSE of test set 2(Hycom):" + " " + str(RMSE_test2_loss))
    print("RMSE of test set 3(dis):" + " " + str(RMSE_test3_loss))

    # alpha_save = np.array(test3_alphas)

    # savemat('alphasavenodis_lambda20.mat', {'alphasave_lambda20': test3_alphas})
    # savemat('dis_outputnodis_lambda20.mat', {'dis_output_lambda20': all_test_outputs3})
    # savemat('dis_clean_input.mat', {'dis_clean_input': dataset[10000:10200, :]})
    # savemat('dis_input.mat', {'dis_input': test3_o_p})
