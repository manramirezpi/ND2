import os
import json
import time
import signal
import logging
import warnings
import numpy as np
import traceback
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from ND2.model import NDformer
from ND2.utils import init_logger, AutoGPU, seed_all
from ND2.search import MCTS
from ND2.GDExpr import GDExpr
from ND2.search.reward_solver import RewardSolver

warnings.filterwarnings("ignore", category=RuntimeWarning)
def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
logger = logging.getLogger('ND2.search')


def main(args):
    # %% Load Data & Init Model
    data = json.load(open(args.data, 'r'))
    for k, v in data.items():
        data[k] = np.array(v)
    data['A'] = data['A'].astype(int)
    data['G'] = data['G'].astype(int)

    # Log Transformation support
    if args.log_target:
        logger.info(f"Target variable '{args.target_var}' is being transformed to Log-space (ln|y|+eps)")
        data[args.target_var] = np.log(np.abs(data[args.target_var]) + 1e-15)

    # init Rewarder
    rewarder = RewardSolver(
        Xv={var: data[var] for var in args.vars},
        Xe={var: data[var] for var in args.edge_vars},
        A=data['A'],
        G=data['G'],
        Y=data[args.target_var],
        mask=None,
    )
    
    # init NDformer
    ndformer = NDformer(device=args.device)
    ndformer.load(args.ndformer_path, weights_only=False)
    ndformer.eval()
    ndformer.set_data(
        Xv={var: data[var] for var in args.vars},
        Xe={var: data[var] for var in args.edge_vars},
        A=data['A'],
        G=data['G'],
        Y=data[args.target_var],
        root_type='node',
        cache_data_emb=True,
    )

    # Operator Pruning
    default_unary = ['neg', 'abs', 'inv', 'exp', 'logabs', 'sin', 'cos', 'tan', 
                     'sqrtabs', 'pow2', 'pow3', 'tanh', 'sigmoid', 'aggr', 'sour', 'targ']
    pruned_unary = [op for op in default_unary if op not in args.prune_ops]
    if len(pruned_unary) != len(default_unary):
        removed = set(default_unary) - set(pruned_unary)
        logger.info(f"Pruned operators: {removed}")

    # init Monte-Carlo Tree Search algorithm
    est = MCTS(
        rewarder=rewarder,
        ndformer=ndformer,
        vars_node=args.vars,
        vars_edge=args.edge_vars,
        unary=pruned_unary,  # Pass the pruned list
        log_per_episode=10,
        log_per_second=None,
        beam_size=args.beam_size,
        use_random_simulate=False,
        max_token_num=args.max_complexity,  # New: limit nested complexity
        max_coeff_num=args.max_coeff,       # New: limit number of constants
    )

    # seed
    if args.initial_expression:
        est.inject_seed(args.initial_expression, visits=100)

    # %% Search
    try:
        est.fit(['node'], episode_limit=args.episodes, time_limit=args.time_limit)
    except KeyboardInterrupt as e: 
        logger.info(f'Interrupted manually.')
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        # Print full Pareto Front
        logger.note("\n" + "="*20 + " FRENTE DE PARETO " + "="*20)
        pareto_candidates = est.Pareto(max_iter=200)
        for prefix, complexity, accuracy in pareto_candidates:
            eq_str = GDExpr.prefix2str(prefix)
            logger.note(f"C:{complexity} | R2:{accuracy:.4f} | Eq: {eq_str}")
        logger.note("="*58 + "\n")

        if est.best_model:
            logger.note(f'Search finished. Discovered model: {GDExpr.prefix2str(est.best_model)}')
            logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in est.best_metric.items()))
        else:
            logger.warning('Search finished without discovering a valid model.')

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'a') as f:
            json.dump(dict(
                host=gethostname(),
                name=args.name,
                seed=args.seed,
                result=est.best_model,
                **est.best_metric,
            ), f)
            f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=f'Search_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('-d', '--device', type=str, default='auto')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--data', type=str, default='./data/synthetic/KUR.json')
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='info')
    parser.add_argument('--ndformer_path', type=str, default='./weights/checkpoint.pth')
    parser.add_argument('--vars', type=str, nargs='*', default=['x', 'omega'])
    parser.add_argument('--edge_vars', type=str, nargs='*', default=[])
    parser.add_argument('--target_var', type=str, default='dx')
    parser.add_argument('--save_path', type=str, default='./result/search.csv')
    parser.add_argument('--episodes', type=int, default=1000000)
    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--time_limit', type=int, default=None)
    parser.add_argument('--initial_expression', type=str, default=None, help='Initial symbolic expression to seed MCTS')
    parser.add_argument('--log_target', action='store_true', help='Apply log-transformation to target variable')
    parser.add_argument('--prune_ops', type=str, nargs='*', default=['sigmoid', 'logabs', 'abs'], 
                        help='Operators to remove from the vocabulary')
    parser.add_argument('--max_complexity', type=int, default=30, help='Maximum number of tokens in the expression')
    parser.add_argument('--max_coeff', type=int, default=5, help='Maximum number of learnable coefficients <C>')
    
    args, unknown = parser.parse_known_args()
    if unknown: 
        warnings.warn(f'Unknown args: {unknown}')
    init_logger(args.name, f'./log/search/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@ZihanYu')
    if args.seed is None: 
        args.seed = np.random.randint(0, 32768)
    seed_all(args.seed)
    if args.device == 'auto': 
        args.device = AutoGPU().choice_gpu(3500, interval=15, force=True)
    logger.info(f'Args: {args}')

    main(args)
