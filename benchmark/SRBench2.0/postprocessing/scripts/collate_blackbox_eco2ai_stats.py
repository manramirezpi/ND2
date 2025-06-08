import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys
from improving_names import *

# Where to load the results
rdir = '../../results_blackbox/'
if len(sys.argv) > 1:
    rdir = sys.argv[1]
else:
    print('no rdir provided, using',rdir)
print('reading results from directory', rdir)

# Where to save the report
sdir = '../../results/black-box/'
if len(sys.argv) > 2:
    sdir = sys.argv[2]
else:
    print('no sdir provided, using', sdir)

print('saving summary to directory', sdir)
if not os.path.exists(sdir):
    print("Creating directory", sdir)
    os.makedirs(sdir)

##########
# load data from json
##########
frames = []
comparison_cols = [
    'dataset',
    'algorithm',
    'random_state',
    'start_time', # so we can find the last one to run
    'experiment_description', # ml method, random seed
    'duration(s)',
    'power_consumption(kWh)',
    'CO2_emissions(kg)',
    'CPU_name',
    'GPU_name',
    'cost'
]

fails = []
import pdb
for f in tqdm(glob(rdir + '/*/*_eco2ai.csv')):
    if 'cv_results' in f: 
        continue

    # leave out symbolic data
    if 'feynman_' in f or 'strogatz_' in f:
        continue

    # leave out LinearReg, Lasso (we have SGD with penalty)
    if any([m in f for m in ['LinearRegression','Lasso','EHCRegressor']]):
        continue

    try: 
        df = pd.read_csv(f, header=0)
        
        if df.shape[0]==0:
            raise Exception("Empty dataframe")
        
        frames.append(df) 
    except Exception as e:
        fails.append([f,e])
        pass
    
print(len(fails),'fails:',fails)

df_results = pd.concat(frames, ignore_index=True)

##########
# cleanup
##########
df_results = df_results.rename(columns={'project_name':'dataset'})
df_results[['algorithm', 'random_state']] = df_results['experiment_description'].str.split(expand=True)
df_results['random_state'] = df_results['random_state'].apply(np.nan_to_num).astype(int)

# df_results = df_results.drop(columns=['id', 'epoch', 'experiment_description'])
df_results = df_results[comparison_cols]

####################
# Improving names
####################
df_results = improve_names(df_results)
df_results = add_metadata(df_results)

# Only SR methods --- excluding sklearn stuff
df_results = df_results[df_results["symbolic_alg"]==True].reset_index(drop=True)

for col in ['algorithm','dataset']:
    print(df_results[col].nunique(), col+'s')

###############################
# save results and summary data
###############################
    
df_results.to_feather(f'{sdir}/power_consumption.feather')
print(f'eco2ai data saved to {sdir}/power_consumption.feather')

########
print('mean trial count:')
print(df_results.groupby('algorithm')['dataset'].count().sort_values()
      / df_results.dataset.nunique())