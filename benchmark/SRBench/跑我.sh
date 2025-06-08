python ./benchmark/SRBench/analyze.py \
    ./benchmark/SRBench/pmlb/strogatz_* \
    -ml nd2-gp,nd2-mcts \
    --local \
    -sym_data \
    -script ./benchmark/SRBench/evaluate_model \
    -results result/srbench \
    -n_jobs 10
