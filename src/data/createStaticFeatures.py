import pandas as pd
import feather
import numpy as np
import os
import sys
import re
import math
import shutil
from scipy import interpolate
import pdb
import datetime
###################################################################################
# June 2020 - preprocess for MTL publication (Jared)
# Sept 2020 - cleaned up for repo construction (Jared)
###################################################################################



#Features to add
# max depth
# log surface area
# latitude
# longitude
# GLM strat perc
# k_d
# SDF

#calculate feature stats

#inital data load, site ids
base_path = "../../data/raw/sb_mtl_data_release/"
obs_df = pd.read_csv(base_path+"obs/temperature_observations.csv")
metadata = pd.read_feather("../../metadata/lake_metadata.feather")
metadata_s = pd.read_csv("../../../lake_conus_surface_temp_2021/metadata/lake_metadata.csv")
metadata.set_index('site_id',inplace=True)
ids = np.unique(obs_df['site_id'].values)
ids = np.array([re.search('nhdhr_(.*)', x).group(1) for x in ids])

n_features = 7

n_lakes = ids.shape[0]

#accumulation data structs for averaging
feat_per_lake = np.zeros((n_lakes,7), dtype=np.float_)
feat_per_lake[:] = np.nan

#calculate averages and std_dev for each input driver across all lakes
for lake_ind, name in enumerate(ids):
    nid = 'nhdhr_' + name
    if metadata_s[metadata_s['site_id']==nid].elevation_m.values.shape[0] == 0:
        pdb.set_trace()
    print("(",lake_ind,"/",str(len(ids)),") ","pre ", name)
    feat_per_lake[lake_ind,0] = metadata.loc[name].max_depth
    feat_per_lake[lake_ind,1] = np.log(metadata.loc[name].surface_area)
    # feat_per_lake[lake_ind,2] = metadata_s[metadata_s['site_id']==nid].elevation_m.values[0]
    feat_per_lake[lake_ind,2] = metadata.loc[name].latitude
    feat_per_lake[lake_ind,3] = metadata.loc[name].longitude
    feat_per_lake[lake_ind,4] = metadata.loc[name].glm_strat_perc
    feat_per_lake[lake_ind,5] = metadata.loc[name].K_d
    feat_per_lake[lake_ind,6] = metadata.loc[name].SDF

mean_per_feat = feat_per_lake.mean(axis=0)
std_per_feat = feat_per_lake.std(axis=0)
norm_feats = (feat_per_lake - mean_per_feat ) / std_per_feat


mean_per_feat = trn_data[:,:,:-1].mean(axis=0)
std_per_feat = trn_data[:,:,:-1].std(axis=0)
norm_feats = (trn_data[:,:,:-1] - mean_per_feat ) / std_per_feat

#add stat feats to existing feats
# for lake_ind, name in enumerate(ids):
#     print("(",lake_ind,"/",str(len(ids)),") "," main ", name)

#     norm_feat_path = "../../data/processed/"+name+"/processed_features.npy"

#     feats = np.load(norm_feat_path)
#     app_arr = np.empty((feats.shape[0],feats.shape[1],7))
#     app_arr[:] = norm_feats[lake_ind]
#     new_feats = np.concatenate((app_arr,feats),axis=2)

#     new_feat_path = "../../data/processed/"+name+"/processed_features_ea"
#     np.save(new_feat_path, new_feats)


#add stat feats to pre-train data
for lake_ind, name in enumerate(ids):
    print("(",lake_ind,"/",str(len(ids)),") "," main ", name)

    norm_feat_path = "../../data/processed/"+name+"/processed_features_pt.npy"
    
    feats = np.load(norm_feat_path)
    app_arr = np.empty((feats.shape[0],feats.shape[1],7))
    app_arr[:] = norm_feats[lake_ind]
    new_feats = np.concatenate((app_arr,feats),axis=2)

    new_feat_path = "../../data/processed/"+name+"/processed_features_pt_ea"
    np.save(new_feat_path, new_feats)
