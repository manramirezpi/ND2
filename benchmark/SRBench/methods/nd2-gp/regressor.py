import torch
import inspect
import numpy as np
import pandas as pd
import ND2 as nd
from ND2.utils import seed_all, init_logger, AutoGPU, AttrDict
from sklearn.base import BaseEstimator, RegressorMixin
from functools import wraps

init_logger('nd2-gp')

eval_kwargs = {
    'scale_x': False,
    'scale_y': False,
}

# 由于 nd.GP 的初始化依赖于 variables，但 SRBench 中只在调用 fit 时才会传入 variables，
# 因此必须在外面包装一层 GP 以缓存输入的参数，直到被调用 fit 时才初始化一个 nd.GP 用于真正的 fit
class GP(nd.GP):
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            raise ValueError(f'Unsupported type: {type(X)}')
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f'Unknown type: {type(X)}')
        variables = [nd.Variable(col, nettype='scalar') for col in X.keys()]
        self.real_est = nd.GP(variables, 
                              binary=self.binary, 
                              unary=self.unary, 
                              log_per_iter=self.log_per_iter,
                              n_iter=self.n_iter,
                              random_state=self.random_state)
        return self.real_est.fit(X, y)

    def predict(self, X):
        return self.real_est.predict(X)

est = GP(
    variables=None,
    binary=[nd.Add, nd.Sub, nd.Mul, nd.Div],
    unary=[nd.Sin, nd.Cos, nd.Neg, nd.Abs],
    log_per_iter=10,
    n_iter=1000,
    random_state=42,
)

def complexity(est):
    return len(est.real_est.eqtree)

def model(est, X=None):
    return est.real_est.eqtree.to_str()
