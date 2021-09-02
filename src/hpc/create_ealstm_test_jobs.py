import re
import pandas as pd
import pdb
import numpy as np
#######################################
# Jan 2021 - Jared - preprocess jobs
#         `  
#######################################
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
test_lakes = ids[~np.isin(ids, train_lakes)]


sbatch = ""
ct = 0
# start = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000]
# end = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,12416]

for i in test_lakes:
    ct += 1
    #for each unique lake
    print(i)

    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#SBATCH --time=00:05:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --gres=gpu:k40:2\n#SBATCH --output=test_ea145_%s.out\n#SBATCH --error=test_ea145_%s.err\n\n#SBATCH -p k40"%(i,i)
    script = "source /home/kumarv/willa099/takeme_evaluate_MTL_tst.sh\n" #cd to directory with training script
    # script2 = "python write_NLDAS_xy_pairs.py %s %s"%(l,l2)
    script2 = "python testEALSTM.py %s"%(i)
    # script2 = "python predict_lakes_EALSTM_COLD_DEBUG.py %s %s"%(l,l2)
    
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch ea145_test_%s.sh"%(i),sbatch])
    with open('../../hpc/ea145_test_{}.sh'.format(i), 'w') as output:
        output.write(all)


compile_job_path= '../../hpc/ea145test_jobs.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)

print(ct, " jobs created, run this to submit: ", compile_job_path)