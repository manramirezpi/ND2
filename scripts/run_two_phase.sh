#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCS22_NJOBS=32  # 占用 32 个核
export PYTHONPATH=.

# 参数网格
repeat=1
SNRs=(-20 -10 0 10 20)
missing_link_ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)
spurious_link_ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)

# 最大并行任务数
MAX_JOBS=1

# 当前后台作业数
function wait_for_jobs {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
}

for repeat_index in $(seq 1 $repeat); do
    for SNR in "${SNRs[@]}"; do
        wait_for_jobs
        echo "Running SNR=${SNR}"
        python ./scripts/run_two_phase.py --data ./data/synthetic/KUR.json --vars x omega \
            --library Polynomial Trigonometric Exponential Fractional CoupledPolynomial CoupledTrigonometric CoupledExponential CoupledFractional \
            --name "Kuramoto_SNR${SNR}" --obs_noise_SNR $SNR &
    done

    for spurious_link_ratio in "${spurious_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running spurious_link_ratio=${spurious_link_ratio}"
        python ./scripts/run_two_phase.py --data ./data/synthetic/KUR.json --vars x omega \
            --library Polynomial Trigonometric Exponential Fractional CoupledPolynomial CoupledTrigonometric CoupledExponential CoupledFractional \
            --name "Kuramoto_spurious_link_ratio${spurious_link_ratio}" --spurious_link_ratio $spurious_link_ratio &
    done

    for missing_link_ratio in "${missing_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running missing_link_ratio=${missing_link_ratio}"
        python ./scripts/run_two_phase.py --data ./data/synthetic/KUR.json --vars x omega \
            --library Polynomial Trigonometric Exponential Fractional CoupledPolynomial CoupledTrigonometric CoupledExponential CoupledFractional \
            --name "Kuramoto_missing_link_ratio${missing_link_ratio}" --missing_link_ratio $missing_link_ratio &
    done

    for SNR in "${SNRs[@]}"; do
        wait_for_jobs
        echo "Running SNR=${SNR}"
        python ./scripts/run_two_phase.py --data ./data/synthetic/KUR.json --vars x omega \
            --name "Kuramoto_SNR${SNR}" --obs_noise_SNR $SNR &
    done

    for spurious_link_ratio in "${spurious_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running spurious_link_ratio=${spurious_link_ratio}"
        python ./scripts/run_two_phase.py --data ./data/synthetic/KUR.json --vars x omega \
            --name "Kuramoto_spurious_link_ratio${spurious_link_ratio}" --spurious_link_ratio $spurious_link_ratio &
    done

    for missing_link_ratio in "${missing_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running missing_link_ratio=${missing_link_ratio}"
        python ./scripts/run_two_phase.py --data ./data/synthetic/KUR.json --vars x omega \
            --name "Kuramoto_missing_link_ratio${missing_link_ratio}" --missing_link_ratio $missing_link_ratio &
    done
done

