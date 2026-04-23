import os
import json
import argparse
import numpy as np
from scipy.special import legendre

def save_dataset(data_node, targets, filename, A=None, G=None):
    if A is None: A = [[0,1],[0,0]]
    if G is None: G = [[0,1]]
    dataset = {"V": len(A), "E": len(G), "A": A, "G": G, "target": targets}
    dataset.update(data_node)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(dataset, f)
    print(f"[OK] Generated: {filename}")

def generate_toy_math():
    """ Baseline dataset: Target = x^2 + 2x """
    np.random.seed(42)
    x_vals = np.random.uniform(-5, 5, 200)
    y_vals = x_vals**2 + 2 * x_vals
    
    # ND2 requires multi-node graph format even for simple 1D data
    data_node = {"x": [[float(x)] for x in x_vals]}
    targets = [[float(y)] for y in y_vals]
    save_dataset(data_node, targets, "data/toy.json", A=[[0]], G=[[0, 0]])

def generate_harmonic():
    """ Harmonic Oscillator: Target(dv) = -x """
    t = np.linspace(0, 10, 200)
    x = np.sin(t)
    dv = -x
    
    data_node = {"x": [[float(v)] for v in x]}
    targets = [[float(v)] for v in dv]
    save_dataset(data_node, targets, "data/harmonic.json", A=[[0]], G=[[0, 0]])

def generate_legendre_recurrence():
    """ Advanced dataset: Phase 7 wide-range recurrence P_{l+1} """
    all_l = list(range(2, 16))
    data_node = {"l_order": [], "x": [], "p_prev1": [], "p_prev2": []}
    targets = []
    
    for l in all_l:
        x_vals = np.cos(np.random.uniform(0, np.pi, 200))
        Pl_plus_1 = legendre(l+1)(x_vals)
        Pl = legendre(l)(x_vals)
        Pl_minus_1 = legendre(l-1)(x_vals)
        
        data_node["l_order"].extend([[0.0, float(l)] for _ in range(len(x_vals))])
        data_node["x"].extend([[0.0, float(xv)] for xv in x_vals])
        data_node["p_prev1"].extend([[0.0, float(p1)] for p1 in Pl])
        data_node["p_prev2"].extend([[0.0, float(p2)] for p2 in Pl_minus_1])
        targets.extend([[0.0, float(pl_plus)] for pl_plus in Pl_plus_1])
        
    save_dataset(data_node, targets, "data/legendre.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "harmonic", "legendre"])
    args = parser.parse_args()
    
    if args.dataset == "toy":
        generate_toy_math()
    elif args.dataset == "harmonic":
        generate_harmonic()
    elif args.dataset == "legendre":
        generate_legendre_recurrence()
