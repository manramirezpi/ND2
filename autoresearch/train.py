import subprocess
import time
import argparse

def run_experiment(dataset_name):
    """
    experiment.py
    ------------
    This is the ONLY file the autonomous agent should modify during a research sprint.
    Tweak the configuration to direct the ND2 MCTS framework towards mathematical discovery.
    """
    
    # --- AGENT CONFIGURATION BLOCK -----------------------------------------
    # Note: ensure `python prepare.py --dataset {dataset_name}` was run first!
    DATA_PATH = f"data/{dataset_name}.json"
    
    # Performance Hyperparameters
    EPISODES = 500
    BEAM_SIZE = 5
    
    # Node features and targets for specific problems
    if dataset_name == "toy":
        VARS_NODE = ["x"]
        TARGET_VAR = "target"
        SEED_EXPR = None
    elif dataset_name == "harmonic":
        VARS_NODE = ["x"]
        TARGET_VAR = "target"
        SEED_EXPR = None
    elif dataset_name == "legendre":
        VARS_NODE = ["l_order", "x", "p_prev1", "p_prev2"]
        TARGET_VAR = "target"
        # Using prior Phase 6 intelligence to warm-start recurrence
        SEED_EXPR = "((x*2.0)*p_prev1)" 
    else:
        raise ValueError(f"Unknown dataset config for {dataset_name}")
    # -----------------------------------------------------------------------
    
    # Build the backend command
    cmd = [
        "python3", "../search.py",
        "--name", f"Experiment_{str(int(time.time()))[-5:]}",
        "--device", "auto",        # Listo para GPU (Auto-detección)
        "--data", DATA_PATH,
        "--ndformer_path", "../weights/checkpoint.pth",
        "--vars", *VARS_NODE,
        "--target_var", TARGET_VAR,
        "--episodes", str(EPISODES),
        "--beam_size", str(BEAM_SIZE),
    ]
    
    if SEED_EXPR:
        cmd.extend(["--initial_expression", SEED_EXPR])
    
    print("\n" + "="*50)
    print(f"🚀 KICKING OFF AUTORESEARCH EXPERIMENT: [{dataset_name.upper()}]")
    print(f"Seed Configuration: {SEED_EXPR if SEED_EXPR else 'Tabula Rasa (None)'}")
    print("="*50 + "\n")
    
    # Trigger the engine
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "harmonic", "legendre"])
    args = parser.parse_args()
    
    run_experiment(args.dataset)
