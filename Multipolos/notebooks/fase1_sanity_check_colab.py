# ============================================================
# Fase 1 — Sanity Check: Monopolo y Dipolo con PySR
# Proyecto: Descubrimiento de Leyes Multipolares (GNN + ND2)
# Equipo: Ingeniería Física — UNAL / Alianza MIT
#
# Instrucciones para Colab:
# 1. Correr la Celda 0 (instalación) — solo la primera vez
# 2. Correr las celdas en orden
# 3. Tiempo esperado por orden multipolar: 2-5 min en GPU
# ============================================================

# ─── CELDA 0: Instalación ────────────────────────────────────────────────────
# (Solo la primera vez en Colab)

# !pip install pysr numpy scipy matplotlib

# from pysr import install
# install()  # Descarga e instala el motor Julia (~2-3 min la primera vez)


# ─── CELDA 1: Generador de datos (copia de multipole_generator.py) ───────────

import numpy as np
import json
from scipy.special import legendre as scipy_legendre
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class GeneratorConfig:
    """Parámetros del generador — controlables por Auto-Research (Optuna)."""
    l_order: int = 1
    Q: float = 1.0
    n_r: int = 50
    n_theta: int = 50
    sampling: str = "grid"
    r_min: float = 0.5
    r_max: float = 5.0
    theta_min: float = 0.01
    theta_max: float = np.pi - 0.01
    noise_snr_db: Optional[float] = None
    seed: int = 42


def compute_potential(r, theta, l, Q=1.0):
    """V_ℓ(r,θ) = Q · P_ℓ(cosθ) / r^(ℓ+1)"""
    Pl = scipy_legendre(l)
    return Q * Pl(np.cos(theta)) / r**(l + 1)


def compute_Er(r, theta, l, Q=1.0):
    """Er = (ℓ+1) · Q · P_ℓ(cosθ) / r^(ℓ+2)"""
    Pl = scipy_legendre(l)
    return (l + 1) * Q * Pl(np.cos(theta)) / r**(l + 2)


def add_noise(signal, snr_db, rng):
    power = np.mean(signal**2)
    noise_std = np.sqrt(power) / 10**(snr_db / 20.0)
    return signal + rng.normal(0, noise_std, size=signal.shape)


def generate_dataset(config: GeneratorConfig):
    rng = np.random.default_rng(config.seed)

    if config.sampling == "grid":
        r_1d = np.linspace(config.r_min, config.r_max, config.n_r)
        t_1d = np.linspace(config.theta_min, config.theta_max, config.n_theta)
        r, theta = np.meshgrid(r_1d, t_1d)
        r, theta = r.flatten(), theta.flatten()
    else:
        n = config.n_r * config.n_theta
        r = rng.uniform(config.r_min, config.r_max, n)
        theta = rng.uniform(config.theta_min, config.theta_max, n)

    V = compute_potential(r, theta, config.l_order, config.Q)
    Er = compute_Er(r, theta, config.l_order, config.Q)

    if config.noise_snr_db is not None:
        V = add_noise(V, config.noise_snr_db, rng)
        Er = add_noise(Er, config.noise_snr_db, rng)

    return r, theta, V, Er


print("✓ Generador de datos cargado.")


# ─── CELDA 2: Visualización de los datos ─────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_multipole(l_order, n_r=60, n_theta=60):
    config = GeneratorConfig(l_order=l_order, n_r=n_r, n_theta=n_theta, sampling="grid")
    r, theta, V, Er = generate_dataset(config)

    names = {0: "Monopolo (ℓ=0)", 1: "Dipolo (ℓ=1)", 2: "Cuadrupolo (ℓ=2)", 3: "Octupolo (ℓ=3)"}
    title = names.get(l_order, f"Multipolo ℓ={l_order}")

    # Reconstruir malla para el plot polar
    r_1d = np.linspace(config.r_min, config.r_max, n_r)
    t_1d = np.linspace(config.theta_min, config.theta_max, n_theta)
    R, T = np.meshgrid(r_1d, t_1d)
    V_grid = V.reshape(n_theta, n_r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Potencial Multipolar — {title}", fontsize=14, fontweight='bold')

    # Plot cartesiano (r vs theta)
    ax1 = axes[0]
    c1 = ax1.contourf(T, R, V_grid, levels=30, cmap='RdBu_r')
    plt.colorbar(c1, ax=ax1, label='V(r,θ)')
    ax1.set_xlabel('θ (radianes)')
    ax1.set_ylabel('r')
    ax1.set_title('Mapa de potencial V(r, θ)')

    # Plot polar
    ax2 = axes[1]
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    c2 = ax2.contourf(T, R, V_grid, levels=30, cmap='RdBu_r')
    plt.colorbar(c2, ax=ax2, label='V(r,θ)')
    ax2.set_title('Vista polar')

    plt.tight_layout()
    plt.show()
    print(f"  Puntos generados: {len(r)}")
    print(f"  Rango de V: [{V.min():.4f}, {V.max():.4f}]")

# Visualizar monopolo, dipolo y cuadrupolo
for l in [0, 1, 2]:
    plot_multipole(l)


# ─── CELDA 3: Sanity Check — Monopolo (ℓ=0) con PySR ────────────────────────
# EXPECTATIVA: PySR debe encontrar exactamente V = Q / r en pocos minutos

from pysr import PySRRegressor

print("=" * 55)
print("FASE 1a — Monopolo (ℓ=0)")
print("Fórmula esperada: V = 1.0 / r")
print("=" * 55)

# Generar datos exactos del monopolo
cfg_mono = GeneratorConfig(l_order=0, n_r=30, n_theta=30, sampling="random", seed=42)
r, theta, V, Er = generate_dataset(cfg_mono)

# Entradas: solo r y cos(θ) — PySR no sabe qué son a priori
X_mono = np.column_stack([r, np.cos(theta)])
y_mono = V

# Configuración PySR — vocabulario mínimo honesto
model_mono = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "inv(x) = 1/x",
        "pow2(x) = x^2",
        "cos_op(x) = cos(x)",   # cos no es nativo en PySR binario, se da así
    ],
    extra_sympy_mappings={
        "inv": lambda x: 1/x,
        "pow2": lambda x: x**2,
        "cos_op": lambda x: x,  # cos(theta) ya viene precalculado como columna
    },
    model_selection="best",
    maxsize=15,          # Limitar complejidad del árbol
    verbosity=1,
    random_state=42,
    deterministic=True,  # Reproducible
    parallelism="serial", # Requerido por deterministic=True
)

model_mono.fit(X_mono, y_mono)

print("\n📊 Frontera de Pareto (Complejidad vs. Error):")
print(model_mono)

print(f"\n🏆 Mejor ecuación encontrada:")
print(f"   {model_mono.sympy()}")

# Validación numérica
V_pred = model_mono.predict(X_mono)
ss_res = np.sum((y_mono - V_pred)**2)
ss_tot = np.sum((y_mono - y_mono.mean())**2)
r2 = 1 - ss_res/ss_tot
print(f"\n   R² = {r2:.6f}  (esperado: 1.000000)")
print(f"   RMSE = {np.sqrt(ss_res/len(y_mono)):.2e}  (esperado: ~0)")


# ─── CELDA 4: Sanity Check — Dipolo (ℓ=1) con PySR ──────────────────────────
# EXPECTATIVA: PySR debe encontrar exactamente V = cos(θ) / r²

print("\n" + "=" * 55)
print("FASE 1b — Dipolo (ℓ=1)")
print("Fórmula esperada: V = cos(θ) / r² = x1 / x0²")
print("  donde x0 = r,  x1 = cos(θ)")
print("=" * 55)

cfg_dip = GeneratorConfig(l_order=1, n_r=40, n_theta=40, sampling="random", seed=42)
r, theta, V, Er = generate_dataset(cfg_dip)

# Entradas: r y cos(θ) — PySR buscará la combinación correcta
X_dip = np.column_stack([r, np.cos(theta)])
y_dip = V

model_dip = PySRRegressor(
    niterations=80,       # Dipolo es más complejo que el monopolo
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "inv(x) = 1/x",
        "pow2(x) = x^2",
        "pow3(x) = x^3",
    ],
    extra_sympy_mappings={
        "inv": lambda x: 1/x,
        "pow2": lambda x: x**2,
        "pow3": lambda x: x**3,
    },
    model_selection="best",
    maxsize=20,
    verbosity=1,
    random_state=42,
    deterministic=True,  # Reproducible
    parallelism="serial", # Requerido por deterministic=True
)

model_dip.fit(X_dip, y_dip)

print("\n📊 Frontera de Pareto (Complejidad vs. Error):")
print(model_dip)

print(f"\n🏆 Mejor ecuación encontrada:")
print(f"   {model_dip.sympy()}")

V_pred_dip = model_dip.predict(X_dip)
ss_res = np.sum((y_dip - V_pred_dip)**2)
ss_tot = np.sum((y_dip - y_dip.mean())**2)
r2_dip = 1 - ss_res/ss_tot
print(f"\n   R² = {r2_dip:.6f}  (esperado: 1.000000)")
print(f"   RMSE = {np.sqrt(ss_res/len(y_dip)):.2e}  (esperado: ~0)")


# ─── CELDA 5: Resumen y Registro de Resultados ───────────────────────────────

import datetime

print("\n" + "=" * 55)
print("RESUMEN DE FASE 1 — SANITY CHECK")
print("=" * 55)
print(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"\n{'Multipolo':<15} {'Ecuación Encontrada':<30} {'R²':>10}")
print("-" * 55)
print(f"{'Monopolo (ℓ=0)':<15} {str(model_mono.sympy())[:30]:<30} {r2:.6f}")
print(f"{'Dipolo (ℓ=1)':<15} {str(model_dip.sympy())[:30]:<30} {r2_dip:.6f}")
print("=" * 55)

# Evaluación de éxito
success_mono = r2 > 0.9999
success_dip = r2_dip > 0.9999

if success_mono and success_dip:
    print("\n✅ SANITY CHECK APROBADO — Listos para Fase 2 (Auto-Research + Cuadrupolo)")
elif success_mono:
    print("\n⚠️  Monopolo encontrado. Dipolo necesita más iteraciones.")
    print("    Sugerencia: aumentar niterations a 150 o ampliar maxsize a 25.")
else:
    print("\n❌ El modelo no convergió. Revisar vocabulario de operadores.")
