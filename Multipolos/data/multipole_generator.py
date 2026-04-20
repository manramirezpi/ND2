"""
multipole_generator.py
======================
Generador de datos sintéticos exactos para la expansión multipolar electromagnética.

Genera valores de potencial V(r, θ) y campo eléctrico Er(r, θ), Eθ(r, θ)
para cualquier orden multipolar ℓ, usando las expresiones analíticas exactas
con polinomios de Legendre de scipy.

Todos los parámetros están centralizados en la sección de configuración
para facilitar la exploración sistemática por el bucle de Auto-Research (Optuna).

Referencia física:
    V_ℓ(r, θ) = (Q_ℓ / r^(ℓ+1)) · P_ℓ(cos θ)
    Er  = (ℓ+1) · Q_ℓ / r^(ℓ+2) · P_ℓ(cos θ)
    Eθ  = Q_ℓ / r^(ℓ+2) · dP_ℓ/dθ · sin θ

Unidades: Sistema Gaussiano simplificado (4πε₀ = 1).
"""

import numpy as np
import json
import os
from scipy.special import legendre as scipy_legendre
from dataclasses import dataclass, asdict
from typing import Optional

# ============================================================
# SECCIÓN DE CONFIGURACIÓN — Parámetros controlables por Auto-Research
# ============================================================

@dataclass
class GeneratorConfig:
    """
    Parámetros del generador. Todos pueden ser modificados por Optuna
    en el bucle de Auto-Research sin tocar el resto del código.
    """

    # --- Orden multipolar ---
    l_order: int = 1                  # Orden ℓ: 0=monopolo, 1=dipolo, 2=cuadrupolo, 3=octupolo...
    Q: float = 1.0                    # Momento multipolar (magnitud de la carga generalizada)

    # --- Resolución de la malla ---
    n_r: int = 50                     # Número de puntos en la dirección radial
    n_theta: int = 50                 # Número de puntos en la dirección angular
    sampling: str = "grid"            # "grid" (malla regular) o "random" (aleatorio uniforme)

    # --- Rango de variables independientes ---
    r_min: float = 0.5               # Radio mínimo (evitar singularidad en r=0)
    r_max: float = 5.0               # Radio máximo
    theta_min: float = 0.01          # Ángulo mínimo en radianes (evitar singularidad en θ=0)
    theta_max: float = np.pi - 0.01  # Ángulo máximo en radianes

    # --- Nivel de ruido (SNR en dB, None = datos exactos sin ruido) ---
    noise_snr_db: Optional[float] = None   # Ej: 40.0 = ruido bajo, 20.0 = ruido moderado, 10.0 = alto

    # --- Semilla de reproducibilidad ---
    seed: int = 42

    # --- Salida ---
    output_dir: str = "synthetic"
    save_json: bool = True            # Si True, guarda en JSON compatible con ND2/PySR
    save_csv: bool = False            # Si True, también guarda en CSV plano


# ============================================================
# FUNCIONES DE CÁLCULO ANALÍTICO EXACTO
# ============================================================

def compute_potential(r: np.ndarray, theta: np.ndarray, l: int, Q: float) -> np.ndarray:
    """
    Calcula el potencial multipolar exacto V_ℓ(r, θ).

    V_ℓ = Q · P_ℓ(cos θ) / r^(ℓ+1)

    Args:
        r      : Array de distancias radiales
        theta  : Array de ángulos polares (en radianes)
        l      : Orden multipolar
        Q      : Momento multipolar

    Returns:
        V : Array de potencial con la misma forma que r y theta
    """
    Pl = scipy_legendre(l)
    cos_theta = np.cos(theta)
    return Q * Pl(cos_theta) / r**(l + 1)


def compute_Er(r: np.ndarray, theta: np.ndarray, l: int, Q: float) -> np.ndarray:
    """
    Componente radial del campo eléctrico.

    Er = (ℓ+1) · Q · P_ℓ(cos θ) / r^(ℓ+2)
    """
    Pl = scipy_legendre(l)
    cos_theta = np.cos(theta)
    return (l + 1) * Q * Pl(cos_theta) / r**(l + 2)


def compute_Etheta(r: np.ndarray, theta: np.ndarray, l: int, Q: float) -> np.ndarray:
    """
    Componente angular del campo eléctrico.

    Eθ = Q · P_ℓ'(cos θ) · sin θ / r^(ℓ+2)

    La derivada dP_ℓ/dθ = -sin(θ) · P_ℓ'(cos θ) se calcula
    numéricamente para robustez con cualquier ℓ.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Derivada numérica de P_ℓ respecto a cos(θ), multiplicada por sin(θ)
    eps = 1e-6
    Pl = scipy_legendre(l)
    dPl_dcos = (Pl(cos_theta + eps) - Pl(cos_theta - eps)) / (2 * eps)
    dPl_dtheta = -sin_theta * dPl_dcos

    return Q * dPl_dtheta / r**(l + 2)


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Añade ruido gaussiano blanco a la señal según el SNR especificado en dB.

    SNR_dB = 20 · log10(||signal|| / ||noise||)

    Args:
        signal : Señal limpia
        snr_db : Relación señal-ruido en dB
        rng    : Generador de números aleatorios (reproducible)

    Returns:
        signal_noisy : Señal con ruido añadido
    """
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 20.0)
    noise_std = np.sqrt(signal_power) / snr_linear
    noise = rng.normal(0, noise_std, size=signal.shape)
    return signal + noise


# ============================================================
# GENERADOR PRINCIPAL
# ============================================================

def generate_dataset(config: GeneratorConfig) -> dict:
    """
    Genera el dataset completo según la configuración dada.

    Returns:
        dataset : Diccionario con arrays de r, theta, V, Er, Etheta
                  y metadatos de la configuración usada.
    """
    rng = np.random.default_rng(config.seed)

    # --- Generación de la malla (r, θ) ---
    if config.sampling == "grid":
        r_1d = np.linspace(config.r_min, config.r_max, config.n_r)
        theta_1d = np.linspace(config.theta_min, config.theta_max, config.n_theta)
        r_grid, theta_grid = np.meshgrid(r_1d, theta_1d)
        r = r_grid.flatten()
        theta = theta_grid.flatten()

    elif config.sampling == "random":
        n_total = config.n_r * config.n_theta
        r = rng.uniform(config.r_min, config.r_max, n_total)
        theta = rng.uniform(config.theta_min, config.theta_max, n_total)

    else:
        raise ValueError(f"Modo de muestreo no reconocido: '{config.sampling}'. Usa 'grid' o 'random'.")

    # --- Cálculo de observables exactos ---
    V = compute_potential(r, theta, config.l_order, config.Q)
    Er = compute_Er(r, theta, config.l_order, config.Q)
    Etheta = compute_Etheta(r, theta, config.l_order, config.Q)

    # --- Aplicar ruido si se especifica ---
    noise_applied = False
    if config.noise_snr_db is not None:
        V = add_noise(V, config.noise_snr_db, rng)
        Er = add_noise(Er, config.noise_snr_db, rng)
        Etheta = add_noise(Etheta, config.noise_snr_db, rng)
        noise_applied = True

    # --- Empaquetar en diccionario ---
    dataset = {
        "metadata": {
            "description": f"Multipolo de orden l={config.l_order} (Q={config.Q})",
            "config": asdict(config),
            "n_samples": len(r),
            "noise_applied": noise_applied,
            "exact_formula": f"V = Q * P_{config.l_order}(cos(theta)) / r^{config.l_order + 1}"
        },
        "inputs": {
            "r": r.tolist(),
            "theta": theta.tolist(),
            "cos_theta": np.cos(theta).tolist(),
        },
        "targets": {
            "V": V.tolist(),
            "Er": Er.tolist(),
            "Etheta": Etheta.tolist(),
        }
    }

    return dataset


def save_dataset(dataset: dict, config: GeneratorConfig) -> str:
    """Guarda el dataset en disco en los formatos especificados."""

    os.makedirs(config.output_dir, exist_ok=True)

    l = config.l_order
    names = {0: "MONOPOLE", 1: "DIPOLE", 2: "QUADRUPOLE", 3: "OCTUPOLE"}
    base_name = names.get(l, f"MULTIPOLE_L{l}")

    suffix = f"_SNR{int(config.noise_snr_db)}dB" if config.noise_snr_db else "_exact"
    filename_base = os.path.join(config.output_dir, f"{base_name}{suffix}")

    saved_paths = []

    if config.save_json:
        path = filename_base + ".json"
        with open(path, "w") as f:
            json.dump(dataset, f, indent=2)
        saved_paths.append(path)
        print(f"  [JSON] Guardado: {path}")

    if config.save_csv:
        import csv
        path = filename_base + ".csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["r", "theta", "cos_theta", "V", "Er", "Etheta"])
            rows = zip(
                dataset["inputs"]["r"],
                dataset["inputs"]["theta"],
                dataset["inputs"]["cos_theta"],
                dataset["targets"]["V"],
                dataset["targets"]["Er"],
                dataset["targets"]["Etheta"],
            )
            writer.writerows(rows)
        saved_paths.append(path)
        print(f"  [CSV]  Guardado: {path}")

    return filename_base


# ============================================================
# EJECUCIÓN DIRECTA — Genera todos los órdenes de referencia
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generador de Datos Multipolares — Proyecto Legendre")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), "synthetic")

    # --- Configuraciones de referencia (sin ruido) ---
    reference_configs = [
        GeneratorConfig(l_order=0, n_r=50, n_theta=50, sampling="grid",
                        noise_snr_db=None, output_dir=output_dir, save_json=True, save_csv=True),
        GeneratorConfig(l_order=1, n_r=50, n_theta=50, sampling="grid",
                        noise_snr_db=None, output_dir=output_dir, save_json=True, save_csv=True),
        GeneratorConfig(l_order=2, n_r=50, n_theta=50, sampling="grid",
                        noise_snr_db=None, output_dir=output_dir, save_json=True, save_csv=True),
        GeneratorConfig(l_order=3, n_r=50, n_theta=50, sampling="grid",
                        noise_snr_db=None, output_dir=output_dir, save_json=True, save_csv=True),
    ]

    # --- Configuraciones con ruido (para Auto-Research) ---
    noisy_configs = [
        GeneratorConfig(l_order=1, n_r=50, n_theta=50, sampling="random",
                        noise_snr_db=40.0, output_dir=output_dir, save_json=True),
        GeneratorConfig(l_order=1, n_r=50, n_theta=50, sampling="random",
                        noise_snr_db=20.0, output_dir=output_dir, save_json=True),
        GeneratorConfig(l_order=1, n_r=50, n_theta=50, sampling="random",
                        noise_snr_db=10.0, output_dir=output_dir, save_json=True),
    ]

    all_configs = reference_configs + noisy_configs

    for cfg in all_configs:
        print(f"\nGenerando: l={cfg.l_order}, muestreo={cfg.sampling}, "
              f"SNR={cfg.noise_snr_db if cfg.noise_snr_db else 'exacto'}")
        dataset = generate_dataset(cfg)
        save_dataset(dataset, cfg)
        n = dataset["metadata"]["n_samples"]
        print(f"  → {n} puntos generados.")

    print("\n" + "=" * 60)
    print("¡Generación completada! Revisa la carpeta 'synthetic/'")
    print("=" * 60)
