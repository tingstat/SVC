###################################################
### Read Me #######################################
## SVC.py: all the estimation and testing functions
## RunSVC.py: the simulation settings and the main function to run the code


import scipy as sp
import numpy as np
from sklearn.utils.extmath import cartesian
from os.path import join, exists
from os import makedirs
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from SVC import svc, MyFancyBar
import sys
import argparse
from numba import jit
from scipy.stats import chi2


######### Estimation Demo Code in simulation ###############
run RunSVC.py --sim 'estimation' --mode 'separate' --'run' 20
## #This command will run the estimation method in the simulation once

######### Testing Demo Code in simulation ###############
run RunSVC.py --sim 'test' --mode 'whole' --'run' 20 --'R' 20
## #This command will run the all the testing procedure for all the settings in the simulation


######### Real Data Demo Code ###############
import pandas as pd
import numpy as np

dir_data = './Data/'
dir_output ='./output'

tract='Left_Hippocampus'
gene = 'Gene_chr2_blcok_96'

gene_select = 'gene_for_left/'+gene+'.csv'

clinical = pd.read_csv(dir_data+'final_clinical_m12_used.csv').drop('RID', axis=1)
nX = np.array(clinical)
nZ =np.array( pd.read_csv(dir_data+gene_select)  )

Y =  np.array( pd.read_csv(dir_data+'Hippocampus_lqd_m_20left.csv') )
nY=np.delete(Y,0,axis=0)



dataset = {'X': nX, 'Y': nY, 'Z': nZ}
m = nY.shape[1]
p = nX.shape[1]

S = np.linspace(0, 1, m).reshape((-1, 1))

dataset['gram_beta'] = [rbf_kernel(S, gamma=1 / 0.01) for _ in range(p) ]
dataset['gram_hz'] = polynomial_kernel(nZ, gamma=1, degree=2, coef0=1)  # h(z, )
dataset['gram_hs'] = rbf_kernel(S, gamma=1 / 0.01)  # h( ,s)

model = svc(dataset, use_banded_rbf=False)

sm_seq = np.logspace(-5, 5, 5)
makedirs(join(dir_output, '{}-{}'.format(tract, gene)), exist_ok=True)


tuning_param = model.tuning(np.ones(p+1), method='BFGS', maxiter=20)
print(tuning_param)


#### Estimate of beta ###########
nb = 50
dense_s = np.linspace(0, 1, nb).reshape((-1, 1))


grams=[rbf_kernel(S, dense_s, gamma=1 / 0.01) for _ in range(p) ]
grams.append(polynomial_kernel(nZ, gamma=1, degree=2, coef0=1))
grams.append(rbf_kernel(S, dense_s, gamma=1 / 0.01))

diag_grams = [1 for _ in range(p + 1)]
beta= model.estimate(grams, diag_grams, components=['beta'],show=True)
np.savetxt(join(dir_output, '{}-{}'.format(tract, gene), 'beta.txt'), beta, fmt='%1.8e')


####### Testing ############
test_res = np.zeros((p + 1, 4))
smooth_seqs = [np.array([1]), sm_seq, sm_seq, sm_seq]
for i in range(p + 1):
    print(i)
    res = model.test(component=i, smooth_seq=smooth_seqs, full_tun_param=tuning_param, show=True)
    test_res[i, :] = np.array(res)
    print('p-value = {}'.format(test_res[i, 3]))
np.savetxt(join(dir_output, '{}-{}'.format(tract, gene), 'test_result.txt'), test_res)
