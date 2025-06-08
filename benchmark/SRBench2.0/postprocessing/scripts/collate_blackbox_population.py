import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys
from improving_names import *

rdir = '../../results_blackbox/'
if len(sys.argv) > 1:
    rdir = sys.argv[1]
else:
    print('no rdir provided, using',rdir)
    
print('reading results from  directory', rdir)
    
##########
# load data from json
##########
frames = []
comparison_cols = [
    'dataset',
    'algorithm',
    'random_state',
    "index",
    "model_size",
    "mse_train",
    "mae_train",
    "r2_train",
    "mse_test",
    "mae_test",
    'r2_test'
]

fails = []
import pdb
for f in tqdm(glob(rdir + '/*/*_population.csv')):
    if 'cv_results' in f: 
        continue

    # leave out symbolic data
    if 'feynman_' in f or 'strogatz_' in f:
        continue

    # leave out LinearReg, Lasso (we have SGD with penalty)
    if any([m in f for m in ['LinearRegression','Lasso','EHCRegressor']]):
        continue
    
    try: 
        frames.append(pd.read_csv(f, header=0)) 
    except Exception as e:
        fails.append([f,e])
        pass
    
print(len(fails),'fails:',fails)

df_results = pd.concat(frames, ignore_index=True)

##########
# cleanup
##########
df_results['random_state'] = df_results['random_state'].astype(int)
df_results = df_results[comparison_cols]

####################
# Improving names
####################
df_results = improve_names(df_results)
df_results = add_metadata(df_results)

for col in ['algorithm','dataset']:
    print(df_results[col].nunique(), col+'s')

###############################
# save results and summary data
###############################
if not os.path.exists('../../results/black-box/'):
    os.makedirs('../../results/black-box/')
    
df_results.to_feather('../../results/black-box/populations.feather')
print('eco2ai data saved to ../../results/black-box/populations.feather')

########
print('mean trial count:')
print(df_results.groupby('algorithm')['dataset'].count().sort_values()
      / df_results.dataset.nunique())