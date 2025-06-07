import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def read_file(filename, label='target', use_dataframe=True, sep=None):
    
    if filename.endswith('gz'):
        compression = 'gzip'
    else:
        compression = None
    
    print('compression:',compression)
    print('filename:',filename)

    input_data = pd.read_csv(filename, sep=sep, compression=compression, engine='python')
    input_data = input_data.fillna(0.0)
     
    # clean up column names
    clean_names = {k:k.strip().replace('.','_').replace('-', '_') for k in input_data.columns}
    if 'class' in clean_names: clean_names['class'] = 'class_'
    input_data = input_data.rename(columns=clean_names)

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1)
    if not use_dataframe:
        X = X.values
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    return X, y, feature_names


