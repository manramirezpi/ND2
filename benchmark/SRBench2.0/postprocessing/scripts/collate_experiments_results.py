"""Collates json-formatted results, cleans them up and saves them as .feather
files.

This works with results generated with evaluate_model and optimize_model.
However, it does not handle results with varying levels of target noise.

There is a special file for loading experiments from the feynman track, 
called collate_groundtruth_results.py.
"""
# Author: William La Cava, williamlacava@gmail.com
# SRBENCH
# License: GPLv3

################################################################################
# Black-box problems
################################################################################
import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys
import pdb
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
    'time_time',
    'model_size',
    'symbolic_model',
    'r2_test',
    'mse_test',
    'mae_test',
    'params'
]

fails = []
for f in tqdm(glob(rdir + '/*/*.json')):
    
    # leave out symbolic data
    if 'feynman_' in f or 'strogatz_' in f:
        continue
    
    try: 
        r = json.load(open(f,'r'))

        if 'cv_results' in f: # cleaning the "tuned" from the algorithm name
            r['algorithm'] = r['algorithm'].replace('tuned', '')

        if isinstance(r['symbolic_model'],list):
#             print(f)
            sm = ['B'+str(i)+'*'+ri for i, ri in enumerate(r['symbolic_model'])]
            sm = '+'.join(sm)
            r['symbolic_model'] = sm
            
        sub_r = {k:v for k,v in r.items() if k in comparison_cols}
    #     df = pd.DataFrame(sub_r)
        frames.append(sub_r) 
    #     print(f)
    #     print(r.keys())
    except Exception as e:
        fails.append([f,e])
        pass
    
print(len(fails),'fails:',fails)

# df_results = pd.concat(frames)
df_results = pd.DataFrame.from_records(frames)
df_results['params_str'] = df_results['params'].apply(str)
df_results = df_results.drop(columns=['params'])

##########
# cleanup
##########
df_results = df_results.rename(columns={'time_time':'training time (s)'})
df_results.loc[:,'training time (hr)'] = df_results['training time (s)']/3600

####################
# Improving names and adding metadata
####################
df_results = improve_names(df_results)
df_results = add_metadata(df_results)

# Only SR methods --- excluding sklearn stuff
df_results = df_results[df_results["symbolic_alg"]==True]

print('mean trial count:')
print(df_results.groupby('algorithm')['dataset'].count().sort_values()
      / df_results.dataset.nunique())

##############################
# Adding more information
##############################
# add modified R2 with 0 floor
df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x,0))

# label friedman ddatasets
df_results.loc[:,'friedman_dataset'] = df_results['dataset'].str.contains('_fri_')
print('loaded',len(df_results),'results')

##############################
# Creating summary report file
##############################

df_results2 = df_results.merge(df_results.groupby('dataset')['algorithm'].nunique().reset_index(),
                              on='dataset',suffixes=('','_count'))

# rankings per trial per dataset
for col in [c for c in df_results2.columns if c.endswith('test') or c.endswith('size')]:
    ascending = 'r2' not in col
    df_results2[col+'_rank_per_trial']=df_results2.groupby(['dataset','random_state'])[col].apply(lambda x: 
                                                                              round(x,3).rank(
                                                                              ascending=ascending))
    
df_sum = df_results2.groupby(['algorithm','dataset'],as_index=False).median()
df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1+x))
df_sum['*algorithm*'] = df_sum.apply(
    lambda row: ('*' if row['symbolic_alg'] else "")+row['algorithm'], axis=1 )
    
df_results = df_results2

# rankings and normalized scores per dataset
for col in [c for c in df_sum.columns if c.endswith('test') or c.endswith('size')]:
    ascending = 'r2' not in col
    df_sum[col+'_rank']=df_sum.groupby(['dataset'])[col].apply(lambda x: 
                                                                        round(x,3).rank(ascending=ascending)
                                                                  )
    df_sum[col+'_norm'] = df_sum.groupby('dataset')[col].apply(lambda x: (x-x.min())/(x.max()-x.min()))
    

for col in ['algorithm','dataset']:
    print(df_results[col].nunique(), col+'s')

###############################
# save results and summary data
###############################

df_sum.to_csv(f'{sdir}/results-summary.csv.gz',
    compression='gzip',index=False)
print(f'summary saved to {sdir}/results-summary.csv.gz')

df_results.to_feather(f'{sdir}/results.feather')
print(f'results saved to {sdir}/results.feather')
