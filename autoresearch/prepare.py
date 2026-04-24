import os
import json
import argparse
import numpy as np
from scipy.special import legendre

def save_dataset(data_node, targets, filename, A=None, G=None, edge_data=None):
    if A is None: A = [[0,1],[0,0]]
    if G is None: G = [[0,1]]
    dataset = {"V": len(A), "E": len(G), "A": A, "G": G, "target": targets}
    dataset.update(data_node)
    if edge_data:
        dataset.update(edge_data)  # Edge features (Xe) stored alongside node features
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(dataset, f)
    print(f"[OK] Generated: {filename}")

def generate_monopole(T=500, Q=1.0):
    """
    Monopolo en el origen con T observadores a distancias aleatorias d.
    Ley exacta: V = Q / d

    Grafo (el primer caso con aristas reales):
      Nodo 0: Monopolo (tiene carga Q)
      Nodo 1: Observador (pasivo, sin propiedades)
      Arista 0->1: lleva la distancia d (Edge Feature)

    Aquí sí usamos --edge_vars d en el comando de búsqueda.
    El MCTS debe descubrir: source(e).Q / edge(e).d
    """
    np.random.seed(42)
    d_vals = np.random.uniform(0.5, 5.0, T)  # distancias aleatorias
    V_vals = Q / d_vals                        # Ley exacta del monopolo

    # === Nodo features: shape (T, V=2) ===
    # Cada fila: [valor_nodo0, valor_nodo1]
    # Nodo 0 = Monopolo (Q constante), Nodo 1 = Observador (Q=0, pasivo)
    Q_node = [[Q, 0.0] for _ in range(T)]

    # === Edge features: shape (T, E=1) ===
    # Cada fila: [valor_arista0] → la distancia d en ese snapshot
    d_edge = [[float(d)] for d in d_vals]

    # === Target: shape (T, V=2) ===
    # El potencial solo se mide en el Observador (nodo 1)
    # El Monopolo no tiene target (ponemos 0.0)
    targets = [[0.0, float(v)] for v in V_vals]

    data_node = {"Q": Q_node}
    edge_data = {"d": d_edge}

    save_dataset(
        data_node, targets,
        filename="data/monopole.json",
        A=[[0, 1], [0, 0]],  # Nodo 0 → Nodo 1 (monopolo → observador)
        G=[[0, 1]],           # 1 arista: del nodo 0 al nodo 1
        edge_data=edge_data
    )

def generate_two_monopoles(T=500):
    """
    2 Monopolos con cargas Q1, Q2 distintas y distancias d1, d2 al observador.
    Ley exacta: V = Q1/d1 + Q2/d2

    Grafo (aquí aggr() es no-trivial):
      Nodo 0: Monopolo_1 (carga Q1)
      Nodo 1: Monopolo_2 (carga Q2)
      Nodo 2: Observador (pasivo)
      Arista 0→2: lleva d1
      Arista 1→2: lleva d2

    El MCTS debe descubrir: aggr(source(e).Q / edge(e).d)
    = Q1/d1 + Q2/d2  ← la suma la hace aggr() automáticamente
    """
    np.random.seed(42)

    # Variamos Q y d para que el motor vea la dependencia real
    Q1_vals = np.random.uniform(0.5, 3.0, T)
    Q2_vals = np.random.uniform(0.5, 3.0, T)
    d1_vals = np.random.uniform(0.5, 5.0, T)
    d2_vals = np.random.uniform(0.5, 5.0, T)
    V_vals  = Q1_vals/d1_vals + Q2_vals/d2_vals   # Ley exacta

    # === Nodo features: shape (T, V=3) ===
    # Cada fila: [Q_nodo0, Q_nodo1, Q_nodo2]
    # Nodo 0 = Monopolo_1, Nodo 1 = Monopolo_2, Nodo 2 = Observador (Q=0)
    Q_node = [[float(q1), float(q2), 0.0]
              for q1, q2 in zip(Q1_vals, Q2_vals)]

    # === Edge features: shape (T, E=2) ===
    # Cada fila: [d_arista0, d_arista1] = [d1, d2]
    d_edge = [[float(d1), float(d2)]
              for d1, d2 in zip(d1_vals, d2_vals)]

    # === Target: shape (T, V=3) ===
    # Solo el Observador (nodo 2) tiene target
    targets = [[0.0, 0.0, float(v)] for v in V_vals]

    data_node = {"Q": Q_node}
    edge_data = {"d": d_edge}

    save_dataset(
        data_node, targets,
        filename="data/two_monopoles.json",
        A=[[0, 0, 1],   # Monopolo_1 → Observador
           [0, 0, 1],   # Monopolo_2 → Observador
           [0, 0, 0]],  # Observador → nadie
        G=[[0, 2],      # Arista 0: Monopolo_1 → Observador
           [1, 2]],     # Arista 1: Monopolo_2 → Observador
        edge_data=edge_data
    )


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
    parser.add_argument("--dataset", type=str, default="toy",
                        choices=["toy", "harmonic", "legendre", "monopole", "two_monopoles"])
    args = parser.parse_args()
    
    if args.dataset == "toy":
        generate_toy_math()
    elif args.dataset == "harmonic":
        generate_harmonic()
    elif args.dataset == "legendre":
        generate_legendre_recurrence()
    elif args.dataset == "monopole":
        generate_monopole()
    elif args.dataset == "two_monopoles":
        generate_two_monopoles()
