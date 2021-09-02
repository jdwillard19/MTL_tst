import numpy as np
import pandas as pd
import pdb
import os
import sys
import re

glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes_wp = np.unique(glm_all_f['target_id'].values) #with prefix

ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]


obs_df = pd.DataFrame()
rmse_df = pd.DataFrame()
for site_id in test_lakes:
	site_obs = pd.read_csv("../../results/ealstm145_results_"+site_id+".csv")
	site_rmse = pd.read_csv("../../results/ealstm145_rmse_"+site_id+".csv")
	obs_df = pd.concat([obs_df, site_obs], ignore_index=True)
	rmse_df = pd.concat([rmse_df, site_rmse], ignore_index=True)
	pdb.set_trace()