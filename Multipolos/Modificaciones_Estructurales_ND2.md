# Modificaciones Estructurales a ND2: Adaptación para Descubrimiento Físico y Multipolos Espaciales

A continuación se documentan las **5 modificaciones arquitectónicas** realizadas al código original de ND2 (`search.py` y dependencias), orientadas a transformar un motor de series temporales (ODE) estricto en un marco flexible de descubrimiento espacial multiparamétrico.

---

### 1. Inyección Dinámica de Características de Nodo (Multi-Variable Node Features)
**Estado Original (ND2 Base):** 
El framework original asumía que el vector de estado principal ($X_v$) contenía un número cerrado de variables dinámicas puramente derivadas de un dataset temporal estático, típicamente limitado a una serie `x` y forzando la evaluación contra derivadas temporales.

**Modificación Actual (`search.py` - `set_data`):** 
Se refactorizó el empaquetado de datos para aceptar un arreglo de longitud arbitraria declarado vía CLI (`--vars`). El código modificado instancia dinámicamente un diccionario combinatorio: `Xv={var: data[var] for var in args.vars}`.

**Impacto (¿Qué permite hacer?):** 
*Habilita N-Dimensionalidad Estática.* Permitió que el motor procesase variables que no son series de tiempo, sino factores físicos independientes simultáneos (e.g. Inyectar `l_order` y los polinomios previos `p_prev1`, `p_prev2` en un solo paso hacia el transformador), posibilitando la formulación en un paso forward recursivo.

---

### 2. Habilitación de Topología Física (Edge Feature Support)
**Estado Original (ND2 Base):** 
El orquestador base invocaba al evaluador de recompensas omitiendo por completo las aristas; las llamadas a `RewardSolver` pasaban por código duro explícitamente `Xe=None`.

**Modificación Actual (`search.py`):** 
Se incluyó el módulo de interconexión agregando la bandera `--edge_vars`. Las inicializaciones de `NDformer` y `RewardSolver` ahora ingieren activamente relaciones topológicas (`Xe={var: data[var] for var in args.edge_vars}`).

**Impacto (¿Qué permite hacer?):** 
*Habilita el Descubrimiento Radial.* Permite mapear distancias (e.g. radios $r$ en configuraciones polares) directamente dentro de las conexiones estructurales del grafo (Aristas), evitando congestionar la información escalar de los nodos (Temperatura, Voltaje) con propiedades netamente espaciales.

---

### 3. Desacoplamiento y Extracción del Frente de Pareto
**Estado Original (ND2 Base):** 
La rutina de búsqueda en MCTS evaluaba múltiples ramas, pero únicamente retornaba y documentaba el (`1`) hiper-modelo dominante al finalizar el ciclo de episodios, ocultando el espectro exploratorio.

**Modificación Actual (`search.py`):** 
Se forzó una extracción de estado post-búsqueda (`est.Pareto()`). Se implementó un bucle y módulo de reporte que desempaqueta, ordena e imprime toda la distribución final de _Complejidad de Ecuación vs Precisión (R²)_.

**Impacto (¿Qué permite hacer?):** 
*Habilita Observabilidad Científica.* Otorga al investigador el poder de analizar formulaciones parciales de las leyes físicas (puntos de inflexión en la frontera Pareto). Fue crucial para descubrir la ley del momento cuadrupolar ($1/r^3$), el cual permaneció "oculto" debajo de soluciones matemáticamente más largas pero menos generalizables para la Física Estándar.

---

### 4. Resolución de Empaquetado en Validación (Tuple Unpacking Fix)
**Estado Original (ND2 Base):** 
Existía un desajuste silencioso de tipos en el núcleo de Monte Carlo (UCB1). Al integrar datos complejos, `solver.evaluate()` arrojaba listas indexadas o tuplas `(reward, modelo)` que el controlador global del MCTS trataba erróneamente de digerir como `floats` escalares continuos, lo que degeneraba en crasheos intermitentes.

**Modificación Actual (`search.py` / Módulo de Recompensas):** 
Se impuso un bloque restrictivo que desempaqueta explícitamente las tuplas en la validación principal de MCTS: `reward, _ = rewarder.evaluate(...)`, curando el estrangulamiento de los tipos (type constraints).

**Impacto (¿Qué permite hacer?):** 
*Habilita Escalabilidad de Evaluación.* Garantizó que el motor de regresión simbólica no colapse de memoria/tipos ante dataset complejos y con mayor peso relacional, estabilizando matemáticamente la ramificación del MCTS para la búsqueda prolongada de los Términos L.

---

### 5. Control Quirúrgico de Anchura Computacional (Beam Size Exposed)
**Estado Original (ND2 Base):** 
La amplitud de retención de exploración paralela (Beam Search Width) en la estructura cíclica de las expresiones aritméticas estaba rígidamente definida dentro del cerebro de MCTS.

**Modificación Actual (`search.py`):** 
Se expuso agresivamente la variable paramétrica `--beam_size` hacia la terminal/cliente.

**Impacto (¿Qué permite hacer?):** 
*Evita el estancamiento en Mínimos Locales Físicos.* Permitió triplicar/cuadruplicar la retención de ramales paralelos, lo cual resultó indispensable a partir de la Fase 3 del entrenamiento de multipolos para forzar descubrimientos no-lineales e intuitivos antes de que el motor de inferencia colapsara a soluciones mediocres como las constantes decimales.
