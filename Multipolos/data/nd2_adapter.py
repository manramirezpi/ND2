import json
import numpy as np
import os

def adapt_to_nd2(r_vals, theta_vals, Y_vals, q_val=1.0, map_r_to='node'):
    """
    Maps spatial configurations to ND2's (T, N, D) format.
    Each spatial sample is a temporal snapshot T.
    """
    num_samples = len(r_vals)
    
    # ND2 Format:
    # Xv: (T, N, D_node)
    # Xe: (T, E, D_edge) (if using edge mapping)
    # A: (N, N)
    # G: List of edge tuples [ (0, 1) ]
    
    final_data = {
        "V": 2,  # Source + Observer
        "E": 1,
        "A": [[0.0, 1.0], [0.0, 0.0]],
        "G": [[0, 1]],
        "q": [[q_val, 0.0] for _ in range(num_samples)],
        "theta": [[0.0, t] for t in theta_vals],
        "target": [[0.0, y] for y in Y_vals]
    }
    
    if map_r_to == 'node':
        final_data["r_node"] = [[0.0, r] for r in r_vals]
    else:
        final_data["r_edge"] = [[r] for r in r_vals]
        
    return final_data

if __name__ == "__main__":
    # Example logic to convert existing dipole/quadrupole data
    pass
