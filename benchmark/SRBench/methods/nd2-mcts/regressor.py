import torch
import inspect
import numpy as np
import pandas as pd
import ND2 as nd
from functools import wraps
from sklearn.base import BaseEstimator, RegressorMixin
from ND2.GDExpr import GDExpr
from ND2.utils import init_logger, AutoGPU


init_logger('nd2-mcts')
device = AutoGPU().choice_gpu(3500, interval=15, force=True)
ndformer_path = './weights/checkpoint.pth'

# 由于 nd.MCTS & nd.RewardCalculator & nd.NDformer 的初始化依赖于 variables，但 SRBench 中只在调用 fit 时才会传入 variables
# 因此必须在外面包装一层 class 以缓存输入的参数，直到被调用 fit 时才初始化这些对象用于真正的 fit
class MCTS(nd.MCTS):
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            raise ValueError(f'Unsupported type: {type(X)}')
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f'Unknown type: {type(X)}')
        variables = list(X.keys()) # [nd.Variable(col, nettype='scalar') for col in X.keys()]

        for var in variables:
            X[var] = X[var][..., np.newaxis]
        y = y[..., np.newaxis]

        A = np.array([[0]])
        G = np.empty((0, 2))

        rewarder = nd.RewardSolver(
            Xv={var: X[var] for var in variables},
            Xe={}, A=A, G=G, Y=y, mask=None,
        )

        ndformer = nd.model.NDformer(device=device)
        ndformer.load(ndformer_path, weights_only=False)
        ndformer.eval()
        ndformer.set_data(
            Xv={var: X[var] for var in variables},
            Xe={}, A=A, G=G, Y=y,
            root_type='node',
            cache_data_emb=True,
        )

        self.real_est = nd.MCTS(
            rewarder=rewarder,
            ndformer=ndformer,
            vars_node=variables,
            vars_edge=[],
            binary=self.binary,
            unary=self.unary,
            log_per_episode=self.log_per_episode,
            log_per_second=self.log_per_second,
            beam_size=self.beam_size,
            use_random_simulate=self.use_random_simulate,
            max_coeff_num=self.max_coeff_num,
        )

        return self.real_est.fit(['node'])

    def predict(self, X):
        if isinstance(X, np.ndarray):
            raise ValueError(f'Unsupported type: {type(X)}')
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f'Unknown type: {type(X)}')
        return GDExpr.eval(self.real_est.best_model, X, [], strict=False)


eval_kwargs = {
    'scale_x': False,
    'scale_y': False,
}


est = MCTS(
    rewarder=None,
    ndformer=None,
    binary=['add', 'sub', 'mul', 'div'],
    unary=['sin', 'cos', 'abs', 'neg'],
    log_per_episode=10,
    log_per_second=None,
    beam_size=10,
    use_random_simulate=False,
    max_coeff_num=5,
)

def complexity(est):
    return len(est.real_est.best_model)

def model(est, X=None):
    return GDExpr.prefix2str(est.real_est.best_model)
