from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_
from datetime import date
import pandas as pd
import pdb
import random
import math
import sys
import re
import os
sys.path.append('../../data')
sys.path.append('../data')
sys.path.append('../../models')
sys.path.append('../models')
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
from torch.utils.data import DataLoader
from pytorch_data_operations import buildCtlstmLakeData
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


#script start
currentDT = datetime.datetime.now()
print(str(currentDT))

####################################################
# (July 2021 - Jared) trains final EA-LSTM model
###################################################

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)



### debug tools
verbose = True
save = True
test = True

lakenames = np.load("lakeList.npy",allow_pickle=True)

#####################3
#params
###########################33
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 15  #number of physical drivers
n_total_feats =n_features
win_shift = 175 #how much to slide the window on training set each time
save = True 
grad_clip = 1.0 #how much to clip the gradient 2-norm in training
dropout = 0.
num_layers = 1
n_hidden = 256
lambda1 = 0


#epoch settings
n_eps = 7000

#load metadata
# metadata = pd.read_csv("../../metadata/lake_metadata.csv")
metadata = pd.read_feather("../../metadata/lake_metadata.feather")

#trim to observed lakes
# metadata = metadata[metadata['num_obs'] > 0]


first_save_epoch = 0
epoch_since_best = 0
yhat_batch_size = 1 #obsolete since this isnt PGDL


final_output_df = pd.DataFrame()


tst_inds = [1, 11, 21, 31, 41]
tst_lakes = np.array([lakenames[i] for i in tst_inds])
trn_lakes = np.delete(lakenames,tst_inds)

if not os.path.exists("./ealstm_trn_data42.npy"):
    (trn_data, _) = buildCtlstmLakeData(trn_lakes,\
                                        seq_length, n_features,\
                                        win_shift = win_shift, begin_loss_ind = begin_loss_ind,\
                                        verbose=True,two_thirds_train=True) 

    np.save("./ealstm_trn_data42.npy",trn_data)
else:
    trn_data = torch.from_numpy(np.load("./ealstm_trn_data42.npy"))

if not os.path.exists("./ealstm_val_data42.npy"):
    (val_data, _) = buildCtlstmLakeData(tst_lakes,\
                                        seq_length, n_features,\
                                        win_shift = win_shift, begin_loss_ind = begin_loss_ind,\
                                        verbose=True, two_thirds_train=True) 

    np.save("./ealstm_val_data42.npy",val_data)
else:
    val_data = torch.from_numpy(np.load("./ealstm_val_data42.npy"))

print("n features: ", trn_data.size()[2]-1)
print("train_data size: ",trn_data.size())
print("val_data size: ",val_data.size())



batch_size = 2000




#Dataset classes
class TemperatureTrainDataset(Dataset):
#training dataset class, allows Dataloader to load both input/target
    def __init__(self, trn_data):
        self.len = trn_data.shape[0]
        self.x_data = trn_data[:,:,:-1].float()
        self.y_data = trn_data[:,:,-1].float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




#format training data for loading
train_data = TemperatureTrainDataset(trn_data)


#format total y-hat data for loading
n_batches = math.floor(trn_data.size()[0] / batch_size)

#batch samplers used to draw samples in dataloaders
batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


#define EA-LSTM class
"""
This file is part of the accompanying code to the manuscript:
Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)
You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""


#define LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = n_total_feats, hidden_size=hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout) #batch_first=True?
        self.out = nn.Linear(hidden_size, 1) #1?
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)))
        if use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden

#method to calculate l1 norm of model
def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    # for name, p in model.named_parameters():
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            #take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val






lstm_net = LSTM(n_features,n_hidden, batch_size)

#tell model to use GPU if needed
if use_gpu:
    lstm_net = lstm_net.cuda()




#define loss and optimizer
mse_criterion = nn.MSELoss()
optimizer = optim.AdamW(lstm_net.parameters())#, weight_decay=0.01)

#training loop

manualSeed = [random.randint(1, 99999999) for i in range(n_eps)]
min_val_rmse = 999
ep_min_mse = -1
patience = 100
#stop training if true
for epoch in range(n_eps):
    # if verbose and epoch % 10 == 0:
    if verbose:
        print("train epoch: ", epoch)

    running_loss = 0.0

    #reload loader for shuffle
    #batch samplers used to draw samples in dataloaders
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)

    trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)


    #zero the parameter gradients
    lstm_net.train(True)
    avg_loss = 0
    batches_done = 0
    ct = 0
    for m, data in enumerate(trainloader, 0):
        #now for mendota data
        #this loop is dated, there is now only one item in testloader

        #parse data into inputs and targets
        inputs = data[0].float()
        targets = data[1].float()
        targets = targets[:, begin_loss_ind:]
        # tmp_dates = tst_dates_target[:, begin_loss_ind:]
        # depths = inputs[:,:,0]


        #cuda commands
        if(use_gpu):
            inputs = inputs.cuda()
            targets = targets.cuda()

        #forward  prop
        lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        # lstm_net.reset_parameters()
        h_state = None
        outputs, h_state = lstm_net(inputs, h_state)
        outputs = outputs.view(outputs.size()[0],-1)

        #calculate losses
        reg1_loss = 0
        if lambda1 > 0:
            reg1_loss = calculate_l1_loss(lstm_net)


        loss_outputs = outputs[:,begin_loss_ind:]
        loss_targets = targets[:,begin_loss_ind:].cpu()


        #get indices to calculate loss
        loss_indices = np.array(np.isfinite(loss_targets.cpu()), dtype='bool_')

        if use_gpu:
            loss_outputs = loss_outputs.cuda()
            loss_targets = loss_targets.cuda()
        loss = mse_criterion(loss_outputs[loss_indices], loss_targets[loss_indices]) + lambda1*reg1_loss 
        #backward

        loss.backward(retain_graph=False)
        if grad_clip > 0:
            clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

        #optimize
        optimizer.step()

        #zero the parameter gradients
        optimizer.zero_grad()
        avg_loss += loss
        batches_done += 1


    if verbose:
        print("train rmse loss=", avg_loss/batches_done)

    with torch.no_grad():
        val_batch_size = val_data.shape[0]
        valloader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False, pin_memory=True)

        mse_criterion = nn.MSELoss()

        lstm_net.eval()
        avg_mse = 0
        mse_ct = 0
        for i, data in enumerate(valloader, 0):
            #parse data into inputs and targets
            inputs = data[:,:,:n_features].float()
            targets = data[:,:,-1].float()
            # targets = targets[:, begin_loss_ind:]
            # tmp_dates = tst_dates[:, begin_loss_ind:]
            # depths = inputs[:,:,0]

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #run model predict
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            pred, h_state = lstm_net(inputs,h_state)

            pred = pred.view(pred.size()[0],-1)
            pred = pred[:, begin_loss_ind:]

            #calculate error
            targets = targets.cpu()
            loss_indices = np.array(np.isfinite(targets.cpu()), dtype='bool_')
            if use_gpu:
                targets = targets.cuda()
            inputs = inputs[:, begin_loss_ind:, :]
            # depths = depths[:, begin_loss_ind:]
            mse = mse_criterion(pred[loss_indices], targets[loss_indices])
            # print("test loss = ",mse)
            mse_ct += 1
            avg_mse += mse

            #append
            avg_mse = avg_mse / mse_ct
            print("test val mse: ",avg_mse)

    if avg_mse < min_val_rmse:
        min_val_rmse = avg_mse
        print("model saved")
        save_path = "../../models/EALSTM_"+str(n_hidden)+"hid_"+str(num_layers)+"_final"
        saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
        ep_since_min = 0
        ep_min_mse = epoch
    else:
        ep_since_min += 1
    print("lowest epoch was ",ep_min_mse," w/ mse=",min_val_rmse)
    if ep_since_min == patience:
        print("training complete")
        break


    # if avg_loss < min_train_rmse:
    #     min_train_rmse = avg_loss
print("model saved")
save_path = "../../models/EALSTM42"
saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)

    # if avg_loss < targ_rmse and epoch > targ_ep:
    #     print("training complete")
    #     break



print("training complete")

