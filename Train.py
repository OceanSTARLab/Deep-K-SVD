import math
import random
from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
# from torchvision import transforms
import time
import Deep_KSVD_For_SSP
from scipy import linalg
import scipy.io as sio
from scipy.io import savemat

datatemp = sio.loadmat('data/ssp_mat.mat')
dataset = np.transpose(datatemp['ssp_mat'])

# disturbance for training, 31 m width, amplitude +-10 m
x = np.linspace(0, 60, 60, dtype=int)
x.shape = (1, 60)
ampli = 10
y1 = ampli * np.sin((np.pi / 18) * x)
y1[:, 18:60] = 0
y2 = -1 * ampli * np.sin((np.pi / 18) * x - 7)
y2[:, 0:5] = 0
y2[:, 23:60] = 0
y3 = ampli * np.sin((np.pi / 18) * x - 14)
y3[:, 0:9] = 0
y3[:, 27:60] = 0
y4 = -1 * ampli * np.sin((np.pi / 18) * x - 21)
y4[:, 0:13] = 0
y4[:, 31:60] = 0
y5 = ampli * np.sin((np.pi / 18) * x - 28)
y5[:, 0:17] = 0
y5[:, 35:60] = 0
y6 = -1 * ampli * np.sin((np.pi / 18) * x - 35)
y6[:, 0:21] = 0
y6[:, 39:60] = 0
y7 = ampli * np.sin((np.pi / 18) * x - 42)
y7[:, 0:25] = 0
y7[:, 43:60] = 0
y8 = -1 * ampli * np.sin((np.pi / 18) * x - 49)
y8[:, 0:29] = 0
y8[:, 47:60] = 0

all_dis = [y1, y2, y3, y4, y5, y6, y7, y8]

# train set
train_origin = dataset[0:9000, :]
train_clean = dataset[0:9000, :]

sigma = 0.9445 # - noise_dev
# To train a model for denoising, noise_dev = 0.9445
# To train a model for disturbance localization, noise_dev = 0.5311
sigma_test = 0.1679
# noise dev. of the test set

train_mean = np.mean(train_origin, axis=0, keepdims=True)
train_demean = train_origin - train_mean  # The perturbation matrix

# test set 1
test1_ori = dataset[9000:10000, :]
test1_mean = np.mean(test1_ori, axis=0, keepdims=True)
test1_demean = test1_ori - test1_mean

train_idx = np.linspace(0, len(train_demean) - 1, num=len(train_demean), dtype=int)
test1_idx = np.linspace(0, len(test1_demean) - 1, num=len(test1_demean), dtype=int)

data_train = []
data_test1 = []

for i in train_idx:
    data_train.append(train_demean[i])
for i in test1_idx:
    data_test1.append(test1_demean[i])

batch_size = len(data_train)

dataloader_test1 = DataLoader(
    data_test1, batch_size=1, shuffle=False, num_workers=0
)
dataloader_train = DataLoader(
    data_train, batch_size=batch_size, shuffle=False, num_workers=0
)

# Create a file to see the output during the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_to_print = open("results_training.csv", "w")
file_to_print.write(str(device) + "\n")
file_to_print.flush()

# Initialization:
input_size = len(data_train[1])
N = 120 # No. of columns of the dictionary

# Dict_init = Deep_KSVD_For_SSP.init_dct(input_size, N)
# Dict_init = Deep_KSVD_For_SSP.init_PCA(np.array(data_train), N)
Dict_init = Deep_KSVD_For_SSP.init_Halfsin(np.array(data_train), N)
DICT_INIT222 = Dict_init.detach().numpy()

Dict_init = Dict_init.to(device)  # Dictionary initialization

c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)
# Initialize parameter c of eq.(10) and (11) in the paper


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

model.to(device)

# Loss function and Optimizer:
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

start = time.time()
epochs = 300
# The number of iterations is usually set to 300
running_loss = 0.0
print_every = 1
train_losses, test_losses, Total_test_loss = [], [], []
test_lam = np.zeros((len(data_test1), 1))
epo = 1



for epoch in range(epochs):
    print("The " + str(epo) + "-th iteration")
    epo += 1
    j = 0
    k = 0
    for i, (x_inputs) in enumerate(dataloader_train, 0):
        # get the input
        x_inputs = (
            x_inputs.to(device),
        )
        # zero the parameter gradients
        optimizer.zero_grad()
        x_inputs = x_inputs[0].to(torch.float32)
        x_inputs_perturbation = x_inputs
        x_inputs_perturbation = x_inputs_perturbation + sigma * (np.random.randn(x_inputs.shape[0], x_inputs.shape[1]))
        x_inputs_perturbation = x_inputs_perturbation + train_mean
        ####################
        # To train a model for disturbance localization, include this part.
        # x_inputs_perturbation[1000:1100, :] = x_inputs_perturbation[1000:1100, :] + y1
        # x_inputs_perturbation[2000:2100, :] = x_inputs_perturbation[2000:2100, :] + y2
        # x_inputs_perturbation[3000:3100, :] = x_inputs_perturbation[3000:3100, :] + y3
        # x_inputs_perturbation[4000:4100, :] = x_inputs_perturbation[4000:4100, :] + y4
        # x_inputs_perturbation[5000:5100, :] = x_inputs_perturbation[5000:5100, :] + y5
        # x_inputs_perturbation[6000:6100, :] = x_inputs_perturbation[6000:6100, :] + y6
        # x_inputs_perturbation[7000:7100, :] = x_inputs_perturbation[7000:7100, :] + y7
        # x_inputs_perturbation[8000:8100, :] = x_inputs_perturbation[8000:8100, :] + y8
        ####################

        x_inputs_perturbation = x_inputs_perturbation - torch.mean(x_inputs_perturbation, axis=0, keepdims=True)
        x_inputs_perturbation = x_inputs_perturbation.to(torch.float32)

        # forward + backward + optimize
        x_outputs, train_alpha, train_lam = model(x_inputs_perturbation, N)
        loss = criterion(x_outputs, x_inputs)  # error
        loss.backward()
        optimizer.step()  # Optimize
        print("\n" + "Training RMSE:" + str(torch.sqrt(loss.data)))

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every - 1:
            # print every x mini-batches
            train_losses.append(running_loss / print_every)
            end = time.time()
            time_curr = end - start
            file_to_print.write("time:" + " " + str(time_curr) + "\n")
            start = time.time()

            with torch.no_grad():
                test_loss = 0
                for test_inputs in dataloader_test1:
                    # Error of test set.
                    # One iteration calculates the RMSE of one SSP
                    test_inputs = (
                        test_inputs.to(device),
                    )
                    test_inputs = test_inputs[0].to(torch.float32)
                    test_inputs_noise = test_inputs
                    test_inputs_noise = test_inputs + sigma_test * np.random.randn(test_inputs.shape[0],
                                                                              test_inputs.shape[1])
                    test_inputs_noise[0].to(torch.float32)
                    test_inputs_noise = (torch.reshape(test_inputs_noise, (1, -1))).float()
                    test_outputs, test_alpha, one_test_lam = model(test_inputs_noise, N)
                    # Test outputs
                    loss = criterion(test_outputs, test_inputs)
                    # error

                    test_lam[k, 0] = one_test_lam
                    test_loss += loss.item()
                    # Add the losses
                    k += 1

                test_loss = test_loss / len(dataloader_test1)
                # Divide the sum of errors by the length of the test set
                RMSE_test_loss = np.sqrt(test_loss)
                test_losses.append(RMSE_test_loss)
                print("RMSE of test set:" + " " + str(RMSE_test_loss))

            end = time.time()
            time_curr = end - start
            file_to_print.write("time:" + " " + str(time_curr) + "\n")

            print("time:" + " " + str(time_curr) + "\n")
            start = time.time()

            s = "[%d, %d] loss_train: %f, loss_test: %f" % (
                epoch + 1,
                (i + 1) * batch_size,
                running_loss / print_every,
                test_loss,
                # sparse_loss,
            )
            s = s + "\n"
            file_to_print.write(s)
            file_to_print.flush()
            running_loss = 0.0

        if i % (10 * print_every) == (10 * print_every) - 1:
            torch.save(model.state_dict(), "model.pth")
            np.savez(
                "losses.npz", train=np.array(test_losses), test=np.array(train_losses)
            )

file_to_print.write("Finished Training")
# savemat('train_res.mat', {'train_res': x_outputs.detach().numpy() - x_inputs.detach().numpy()})
# savemat('train_lam.mat', {'train_lam': train_lam.detach().numpy()})
# savemat('test_lam.mat', {'test_lam' : test_lam})

print("Finished Training, Dictionary has been updated")

torch.save(model.state_dict(), "model.pth")
print("Finished Training, Model has been saved")
plt.figure()
plt.title('Test RMSE in the training process')
plt.plot(np.linspace(0, epochs, epochs), test_losses)
plt.show()
