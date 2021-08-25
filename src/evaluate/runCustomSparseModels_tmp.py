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
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')
from pytorch_data_operations import buildLakeDataForRNNPretrain, calculate_energy,calculate_ec_loss_manylakes, transformTempToDensity, calculate_dc_loss
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
#multiple dataloader wrapping?
import pdb
from torch.utils.data import DataLoader
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs

####################################################################################################3
# (Sept 2020 - Jared) - this script runs all the sparse PGDL models for target lakes and records RMSE
###########################################################################################################


#script start
currentDT = datetime.datetime.now()
print(str(currentDT))

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)
# sources = pd.read_csv('pgdtl_rmse_pball_sources.csv')
ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")

train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
test_lakes = ids[~np.isin(ids, train_lakes)]


#to iterate through
n_profiles = [1,2,5,10,15,20,25,30,35,40,45,50]
seeds = [0,1,2,3,4]


#monitor which sites do not have "x" observations (no_x below)
no_50 = []
no_45 = []
no_40 = []
no_35 = []
no_30 = []
no_25 = []

### debug tools
debug_train = False
debug_end = False
verbose = False
pretrain = False
save = True
save_pretrain = True

#####################3
#params
###########################33

unsup_loss_cutoff = 40
dc_unsup_loss_cutoff = 1e-3
dc_unsup_loss_cutoff2 = 1e-2
n_hidden = 20 #fixed
train_epochs = 10000
pretrain_epochs = 10000


n_ep = pretrain_epochs  #number of epochs

if debug_train or debug_end:
    n_ep = 10
first_save_epoch = 0
patience = 100

#ow
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 8  #number of physical drivers
win_shift = 175 #how much to slide the window on training set each time
save = True 

#for each test lake
for lake_ct, lakename in enumerate(test_lakes):
    target_id = lakename
    print("(",lake_ct,"/",len(test_lakes),"): ", lakename)
    data_dir = "../../data/processed/"+lakename+"/"

    for n_prof in n_profiles:
        for seed_ct, seed in enumerate(seeds):
            if not os.path.exists("../../models/"+lakename+"/PGRNN_sparse_" + str(n_prof) + "_" + str(seed)):
                print('not enough observations')
                if n_prof == 25:
                    no_25.append(target_id)
                    continue
                elif n_prof == 30:
                    no_30.append(target_id)
                    continue
                elif n_prof == 35:
                    no_35.append(target_id)
                    continue
                elif n_prof == 40:
                    no_40.append(target_id)
                    continue
                elif n_prof == 45:
                    no_45.append(target_id)
                    continue
                elif n_prof == 50:
                    no_50.append(target_id)
                    continue
                else:
                    continue
            # load_path = "../../models/"+lakename+"/PGRNN_sparse_" + str(n_prof) + "_" + str(seed)

            # ###############################
            # # data preprocess
            # ##################################
            # #create train and test sets

            # (trn_data, _, _, _, _, _, _, _,
            # _) = buildLakeDataForRNN_manylakes_finetune2(lakename, data_dir, seq_length, n_features,
            #                                    win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
            #                                    outputFullTestMatrix=False, 
            #                                    allTestSeq=False, sparseCustom=n_prof, randomSeed=seed) 
            # #if error code is returned (trn data as int), skip and record id
            # print(trn_data)
            # if isinstance(trn_data, int):
            #     pdb.set_trace()
            #     target_id = lakename
            #     if trn_data == 25:
            #         no_25.append(target_id)
            #         continue
            #     elif trn_data == 30:
            #         no_30.append(target_id)
            #         continue
            #     elif trn_data == 35:
            #         no_35.append(target_id)
            #         continue
            #     elif trn_data == 40:
            #         no_40.append(target_id)
            #         continue
            #     elif trn_data == 45:
            #         no_45.append(target_id)
            #         continue
            #     elif trn_data == 50:
            #         no_50.append(target_id)
            #         continue
            
print("25-50 missed")
# print(len(no_25),no_25)
print(len(no_25))
print(len(no_30))
# print(len(no_30),no_30)
# print(len(no_35),no_35)
print(len(no_35))
# print(len(no_40),no_40)
print(len(no_40))
print(len(no_45))
# print(len(no_45),no_45)
# print(len(no_50),no_50)
print(len(no_50))
