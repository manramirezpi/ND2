# Modificaciones Estructurales a ND2: Adaptación para Descubrimiento Físico y Multipolos Espaciales

A continuación se documentan las **5 modificaciones principales** realizadas al código original de ND2 (`search.py` y dependencias). El objetivo fue transformar un programa diseñado inicialmente para descubrir ecuaciones de series de tiempo, en un sistema más versátil capaz de descubrir leyes de la física en el espacio tridimensional (como el decaimiento de las fuerzas electromagnéticas).

---

### 1. Soporte para Múltiples Variables de Entrada

**Estado Original:**
El programa original asumía que solo se le entregaría una variable matemática simple (usualmente el tiempo) para descubrir derivadas directas.

**Modificación Actual:**
Reescribimos la forma en que el sistema carga los datos. Ahora el sistema puede ingerir múltiples variables y "propiedades" al mismo tiempo. Por ejemplo, podemos alimentar al algoritmo con el número de orden cuántico ($L$), la posición angular ($x$), y polinomios matemáticos previos, todo en paralelo.

**¿Por qué es importante?**
Permite que la red neuronal analice sistemas de la vida real que dependen de muchos factores simultáneos, lo cual es obligatorio para encontrar ecuaciones como la ley de recurrencia de Legendre.

---

### 2. Soporte para Relaciones de Distancia Espacial

**Estado Original:**
Originalmente, el código ignoraba por diseño las "aristas" o conexiones entre diferentes puntos de datos, tratándolas simplemente como vacías o inexistentes.

**Modificación Actual:**
Habilitamos paramétricamente la carga de Atributos de Arista (Edge Features) en toda la arquitectura del programa.

**¿Por qué es importante?**
Esto le enseñó al programa el concepto de "distancia". Es lo que permitió mapear explícitamente el radio ($r$) que separa dos cuerpos en el espacio. Sin esto, el motor sería ciego a la caída de las fuerzas físicas según se aleja la distancia.

---

### 3. Visibilidad Total de Ecuaciones (Frente de Pareto)

**Estado Original:**
Al terminar de procesar durante horas, el programa originalmente imprimía únicamente una (`1`) ecuación ganadora, ocultando miles de ecuaciones intermedias.

**Modificación Actual:**
Hackeamos la salida del programa para que imprima toda la lista ordenada de ecuaciones descubiertas (desde la más simple hasta la más absurdamente compleja). A esta lista se le conoce como el Frente de Pareto.

**¿Por qué es importante?**
Las máquinas a veces eligen fórmulas matemáticas muy largas que logran precisión "falsa" añadiendo constantes inútiles. Al ver toda la lista, nosotros como físicos pudimos detectar cuándo la máquina había encontrado correctamente nuestra Ley Cuadrupolar ($1/r^3$) aunque la máquina no la hubiera clasificado como la mejor opción absoluta.

---

### 4. Corrección de Errores de Lectura Numérica

**Estado Original:**
Había un error silente en el núcleo matemático predictivo del programa. Cuando se evaluaban datasets físicos muy pesados, el sistema interno recibía una "caja de datos" (tuplas) en vez de un número simple, lo cual provocaba que la ejecución fallara o se "crasheara".

**Modificación Actual:**
Se reprogramó la función de desempaquetado de evaluaciones para garantizar que el motor estadístico reciba siempre el número exacto del puntaje.

**¿Por qué es importante?**
Evita que el programa colapse. Gracias a esto, hoy podemos dejar el motor buscando ecuaciones extremadamente exóticas durante toda la noche sin riesgo de interrupciones técnicas.

---

### 5. Control Quirúrgico de Amplitud de Exploración (Beam Size)

**Estado Original:**
Haciendo una analogía: El número de ecuaciones que la Inteligencia Artificial podía "mantener en su cabeza al mismo tiempo" estaba bloqueado a un número pequeño de fábrica.

**Modificación Actual:**
Añadimos un comando para que el usuario pueda aumentar multiplicar a voluntad ese límite de "memoria imaginativa" paralela.

**¿Por qué es importante?**
Previene que el sistema se quede atascado intentando "mejorar" ecuaciones defectuosas simplemente añadiéndoles decimales (como probar $x * 2.0001$). Al ensanchar la búsqueda, el algoritmo puede descartar fórmulas malas a tiempo y saltar hacia una formulación matemática mucho más abstracta y correcta.
