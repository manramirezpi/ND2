# Hoja de Ruta: Descubrimiento de Leyes de Expansión Multipolar
## Estrategia: Graph Neural Networks + Regresión Simbólica (GNN + ND2/PySR)
### Equipo: Ingeniería Física — UNAL / Alianza MIT

---

## Objetivo Final

Descubrir de forma autónoma, a partir de datos sintéticos, las leyes que rigen la expansión
multipolar electromagnética — incluyendo la expresión de recurrencia de los polinomios de
Legendre — usando el pipeline GNN + Regresión Simbólica, con miras a transferir la
metodología ganadora a sistemas nucleares desconocidos.

---

## Principios de Diseño del Experimento

1. **Transparencia metodológica**: El modelo recibe solo datos y un vocabulario de operadores
   matemáticos elementales. No se le entrega ninguna estructura funcional del dominio
   (no `legendre(n,x)`, no `assoc_legendre`, no la relación de recurrencia como operador).

2. **Vocabulario mínimo honesto**: Operadores justificables para cualquier físico sin
   conocimiento previo del problema:
   - Binarios: `+`, `-`, `*`, `/`
   - Unarios: `cos`, `sin`, `exp`, `log`, `pow2`, `pow3`, `inv` (= 1/x)
   - Extensión en Julia/PySR: `pow_neg2` (1/x²), `pow_neg3` (1/x³)

3. **Escalabilidad como métrica**: La arquitectura GNN no debe modificarse al pasar de
   2 cargas (dipolo) a N cargas (multipolo general). Esa invarianza es el argumento
   científico principal frente a los otros equipos.

4. **Transferibilidad**: La regla simbólica descubierta (no los pesos de la red) es el
   resultado transferible. Una caja negra no es un resultado publicable.

---

## Arquitectura del Pipeline

```
NIVEL 1: Problema de Campo (Muchas Cargas)
─────────────────────────────────────────
Datos (posiciones de cargas, puntos de observación, V medido)
              ↓
        [GNN - Grafo Bipartito]
    Nodos fuente: cargas (q_i, posición_i)
    Nodos observador: puntos de medición
    Aristas: distancia r_ij y ángulo θ_ij como atributos
              ↓
    Representación latente del campo por punto
              ↓
        [PySR / ND2]
    Busca fórmula simbólica: V = f(r, θ)
              ↓
    Resultado: V ≈ Q·P_ℓ(cosθ) / r^(ℓ+1)

NIVEL 2: Problema de Recurrencia (Regla Generativa)
────────────────────────────────────────────────────
Datos de entrenamiento: tripletas (P_{ℓ-1}(x), P_ℓ(x), P_{ℓ+1}(x)) para x ∈ [-1,1]
No necesita GNN. Entrada directa a PySR/ND2.
              ↓
        [PySR / ND2]
    Busca: P_{ℓ+1} = g(P_ℓ, P_{ℓ-1}, x, ℓ)
              ↓
    Resultado esperado: P_{ℓ+1} = ((2ℓ+1)·x·P_ℓ - ℓ·P_{ℓ-1}) / (ℓ+1)
    Complejidad del árbol: ~13 nodos (verificado experimentalmente)
```

---

## Fases de Implementación

### FASE 0 — Generador de Datos Sintéticos *(Semana 1)*

**Objetivo**: Script Python independiente de cualquier framework ML que genere datos
exactos para cualquier orden ℓ. Este script es el "Apéndice A" del paper.

**Entregables**:
- `data/multipole_generator.py`: Genera tripletas (r, θ, V) usando `scipy.special.legendre`
- `data/recurrence_dataset.py`: Genera tripletas (P_{ℓ-1}, P_ℓ, P_{ℓ+1}) para ℓ = 1..N
- Datasets guardados en `data/synthetic/`: `MONOPOLE.json`, `DIPOLE.json`, `QUADRUPOLE.json`

**Validación**: Verificar numéricamente que los datos generados satisfacen la ecuación
exacta con error < 1e-10.

---

### FASE 1 — Sanity Check: Monopolo (ℓ=0) *(Semana 1-2)*

**Objetivo**: Verificar que PySR/ND2 puede descubrir V = Q/r.

Si no puede descubrir el caso más simple (una sola potencia de r), no tiene sentido
continuar con los órdenes superiores.

**Configuración PySR (Julia)**:
```julia
model = PySRRegressor(
    niterations=50,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["inv(x) = 1/x"],
    model_selection="best"
)
```

**Métrica de éxito**: El modelo reporta exactamente `Q / r` con R² = 1.0000.

---

### FASE 2 — Dipolo (ℓ=1) con Auto-Research *(Semana 2-3)*

**Objetivo**: Descubrir V = Q·cos(θ)/r². Aquí entra el Auto-Research.

**El Auto-Research** responde de forma sistemática:
- ¿Cuántos puntos de datos son suficientes? (Búsqueda: N ∈ {500, 1000, 5000, 10000})
- ¿Qué nivel de ruido tolera el modelo? (SNR ∈ {∞, 40dB, 20dB, 10dB})
- ¿`pow_neg2` mejora o `inv` + `mul` es suficiente?

**Implementación del Auto-Research**:
```python
import optuna

def objective(trial):
    n_points = trial.suggest_categorical('n_points', [500, 1000, 5000])
    noise_level = trial.suggest_float('noise_level', 0.0, 0.1)
    use_pow_neg2 = trial.suggest_categorical('use_pow_neg2', [True, False])
    
    # Generar datos con esos parámetros
    # Correr PySR por exactamente 5 minutos (GPU) o 20 minutos (CPU)
    # Retornar R² del mejor modelo encontrado
    return r2_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Resultado esperado del Auto-Research**: Una tabla que muestra la configuración mínima
para recuperar exactamente `cos(θ) / r²`.

---

### FASE 3 — Cuadrupolo (ℓ=2) y Escalado *(Semana 3-4)*

**Objetivo**: Descubrir V ∝ (3cos²θ - 1)/r³. La GNN toma protagonismo aquí.

**El argumento de escalado** (ventaja frente a otros equipos):
Usar exactamente el mismo código de Fase 2 pero con 4 cargas en el grafo en lugar de 2.
Si el tiempo de cómputo no escala con el número de cargas, el argumento está probado.

**Benchmark**: Registrar tiempo de cómputo para ℓ = 1, 2, 3 con GNN vs PySR puro.
Si GNN escala mejor, ese es el gráfico principal del paper.

---

### FASE 4 — Relación de Recurrencia *(Semana 4-5)*

**Objetivo**: Descubrir P_{ℓ+1} = ((2ℓ+1)·x·P_ℓ - ℓ·P_{ℓ-1}) / (ℓ+1)

**Por qué es el resultado más importante**:
- Complejidad del árbol: ~13 nodos (manejable por PySR)
- Una vez descubierta, calcula P_{1024} con un simple loop en microsegundos
- Es directamente transferible: la misma estrategia busca la recurrencia de cualquier
  familia de funciones en un sistema nuclear desconocido

**Formulación honesta del problema**:
- Input: (P_{ℓ-1}(x), P_ℓ(x), ℓ, x) para combinaciones de ℓ ∈ {1..20} y x ∈ [-1,1]
- Output: P_{ℓ+1}(x)
- ℓ es el índice natural de la familia, no una "pseudo-temporalidad"

**Validación**: Con la regla descubierta, calcular P_{64} y comparar con `scipy.special.legendre(64)`.

---

### FASE 5 — Límites y Conclusiones del Método *(Semana 5-6)*

**Objetivo**: Caracterizar honestamente los límites de GNN + ND2 para este problema.

**Preguntas de investigación**:
- ¿Hasta qué orden ℓ puede recuperar la solución algebraica directamente?
- ¿A partir de qué SNR (nivel de ruido) el método falla?
- ¿La recurrencia se puede descubrir sin conocer ℓ como variable de entrada?
- ¿La Ecuación Diferencial de Legendre es alcanzable? (Muro conocido: requiere operadores
  de derivada que PySR/ND2 no tienen nativamente → conclusión honesta del paper)

---

## Estrategia de Extensión a Sistemas Nucleares

Una vez validado en el sistema de juguete (Legendre), la metodología se transfiere así:

1. **Medir observables** del sistema nuclear para diferentes configuraciones (rol del experimentalista)
2. **Construir el grafo**: núcleos o nucleones como nodos, interacciones como aristas con
   atributos físicos (distancia de separación, número másico, spin)
3. **GNN extrae** la representación del observable en función de la topología nuclear
4. **PySR/ND2 busca** la ley simbólica con el mismo vocabulario de operadores elementales
5. **Buscar la recurrencia**: ¿Existe una regla generativa entre diferentes configuraciones
   del sistema nuclear análoga a la de Legendre?

---

## Stack Tecnológico

| Componente | Herramienta | Justificación |
|---|---|---|
| Motor de búsqueda simbólica | PySR + Julia | Gradientes analíticos, benchmarking reproducible, Frontera de Pareto |
| GNN para topología | PyTorch Geometric (PyG) | Grafo bipartito flexible, escalable |
| Auto-Research | Optuna (Python) | Búsqueda Bayesiana, integración simple |
| Generador de datos | NumPy + SciPy | Independiente de ML, transparente |
| Entorno de ejecución | Google Colab (GPU) | Experimentos de 5 min por trial |
| Control de experimentos | MLflow o W&B | Registro reproducible de cada trial |

---

## Estructura de Archivos del Proyecto

```
ND2/
├── ROADMAP_multipolos.md          ← Este documento
├── data/
│   ├── multipole_generator.py     ← Fase 0: Generador de datos
│   ├── recurrence_dataset.py      ← Fase 4: Dataset de recurrencia
│   └── synthetic/
│       ├── MONOPOLE.json
│       ├── DIPOLE.json
│       └── QUADRUPOLE.json
├── models/
│   ├── gnn_bipartite.py           ← GNN bipartita (Fase 1-3)
│   └── pysr_runner.py             ← Wrapper PySR con vocabulario configurable
├── auto_research/
│   └── optuna_search.py           ← Bucle de Auto-Research (Fase 2)
├── results/
│   └── pareto_curves/             ← Fronteras complejidad vs. error
└── notebooks/
    ├── 01_monopole_sanity.ipynb
    ├── 02_dipole_autoresearch.ipynb
    ├── 03_quadrupole_scaling.ipynb
    └── 04_recurrence_discovery.ipynb
```

---

## Métricas de Evaluación (Para el Paper)

1. **Error de reconstrucción**: `L = ||V_pred - V_exact||² / ||V_exact||²`
2. **Exactitud simbólica**: ¿Se recupera la forma funcional correcta? (Binario: sí/no)
3. **Complejidad del árbol**: Número de nodos en la expresión encontrada
4. **Robustez al ruido**: Degradación de exactitud simbólica vs. SNR decreciente
5. **Escalado**: Tiempo de cómputo vs. número de cargas N (GNN vs. PySR puro)
6. **Costo computacional**: Número de trials de Auto-Research para convergencia

---

*Documento de trabajo interno. Proyecto en colaboración UNAL / MIT.*
*Última actualización: Abril 2026*
