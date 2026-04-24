# Modificaciones estructurales a ND2: Adaptación para descubrimiento físico y multipolos espaciales

A continuación se documentan las **5 modificaciones principales** realizadas al código original de ND2 (`search.py` y dependencias). El objetivo fue transformar un programa diseñado inicialmente para descubrir ecuaciones de series de tiempo, en un sistema más versátil capaz de descubrir leyes de la física en el espacio tridimensional (como el decaimiento multipolar).

---

### 1. Soporte para múltiples variables simultáneas

**Estado original:**
El programa asumía un bloque rígido de variables, generalmente limitado a observar cómo una característica $x$ variaba en el tiempo.

**Modificación actual:**
Reescribimos la ingestión de datos en el archivo principal para que construya dinámicamente el vector espacial usando todos los nombres de array que dictemos por consola (`--vars`).

```python
# Código implementado en search.py
ndformer.set_data(
    Xv={var: data[var] for var in args.vars}, # Ej. args.vars = ['l_order', 'x', 'p_prev1']
    ...
)
```
---

### 2. Soporte para relaciones de distancia espacial (Aristas)

**Estado original:**
La topología subyacente del grafo (`Xe`) era ignorada e introducida explícitamente como "nula" o vacía.

**Modificación actual:**
Habilitamos paramétricamente la carga de 'Edge Features' para que informen las conexiones entre los nodos físicos a la matriz de atención de la IA.

```python
# Código implementado en search.py
rewarder = RewardSolver(
    Xv={var: data[var] for var in args.vars},
    Xe={var: data[var] for var in args.edge_vars}, # Permite mapear el radio r o vector distancia
    ...
)
```
---

### 3. Visibilidad total de ecuaciones (Frente de Pareto)

**Estado original:**
Al finalizar las iteraciones del árbol de Monte Carlo, el script base descartaba la historia de la búsqueda e imprimía solo al único "ganador" pre-calculado.

**Modificación actual:**
Hackeamos la salida para extraer y organizar el diccionario completo de las mejores ecuaciones compitiendo por nivel de longitud.

```python
# Código implementado en search.py post-búsqueda
print("==================== FRENTE DE PARETO ====================")
front = est.Pareto() # Fuerzo extracción del historial estructural de Monte Carlo
for eq in front:
    print(f"Complex:{eq[1]} | R2:{eq[0]:.4f} | Eq: {GDExpr.prefix2str(eq[2])}")
```

**¿Por qué es importante?**
Las IA comúnmente encuentran modelos sobreajustados (fórmulas largas llenas de decimales innecesarios con R²=1.0). Obligarla a imprimir el frente de Pareto nos permite detectar la maravilla de la Ley Cuadrupolar exacta $1/r^3$ antes de que el motor la descartara en favor de una ecuación espuria.

---

### 4. Corrección de errores de lectura numérica (Tuple Unpack Fix)

**Estado original:**
El núcleo UCB1 de Monte Carlo tenía limitaciones de tipo variables. Ocasionalmente, los validadores retornaban tuplas o diccionarios complejos y la máquina esperaba un flotante, causando bloqueos aleatorios al evaluar ramas largas.

**Modificación actual:**
Impusimos el desempaquetado exacto que aísla solo la recompensa estocástica, descartando los rastros estructurales que estrangulaban a la capa recursiva.

```python
# Código implementado en la sincronización MCTS
reward, _ = rewarder.evaluate(prefix_with_coef, {}) # Extracción forzada del score flotante
# Reemplaza la fallida asignación directa que desencadenaba el error "dictionary cannot be float"
```

**¿Por qué es importante?**
Concurrió en la robustez. Resolvió el "crasheo" en maratones de entrenamiento y previno reinicios innecesarios en la nube.

---

### 5. Control de amplitud de exploración (Beam Size)

**Estado original:**
El grado de variables algebraicas que la inteligencia artificial procesaba a la vez ("imaginación" paralela) estaba congelado a un valor empírico predeterminado.

**Modificación actual:**
Exposición de la variable en el entorno superior para dotar al operador heurístico de control sobre la anchura de las ramas.

```python
# Código implementado en search.py
parser.add_argument('--beam_size', type=int, default=20) 

# Transferido a MCTS setup
est = MCTS(..., beam_size=args.beam_size)
```

**¿Por qué es importante?**
En las fases elevadas (como `l=5` o inducciones largas), el tamaño por defecto resultaba en estancamiento dentro de mínimos locales. Aumentar drásticamente el `beam_size` facilitó un salto deductivo hacia una ramificación abstracta y matemáticamente elegante sin sobreajustarse.
