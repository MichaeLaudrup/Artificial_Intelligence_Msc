### **4. Ingeniería de Características I: Transformación Temporal**

Para alimentar la arquitectura híbrida descrita en los objetivos, es necesario transformar los datos tabulares de interacción en **series temporales estructuradas**. Esta representación sirve a un doble propósito metodológico:

1.  **Entrada para el Modelo Supervisado (Transformer):** Permite capturar dependencias secuenciales a largo plazo (Objetivo 2.2).
2.  **Base para el Modelo No Supervisado (Clustering):** Proporciona los vectores de comportamiento dinámico necesarios para identificar arquetipos de estudiantes (Objetivo 2.1).

A diferencia de los enfoques tradicionales que agregan la actividad en una única métrica estática (*total clics*), esta fase descompone el comportamiento del estudiante en una secuencia de pasos semanales. Esto permite al modelo capturar la **dinámica evolutiva del aprendizaje**, diferenciando entre un estudiante constante y uno que abandona progresivamente.

**Estrategia de Modelado Temporal:**
Se ha definido una ventana de observación que cubre el ciclo de vida completo del curso, aplicando una estrategia de *Bucketing* para gestionar los extremos temporales sin perder información:

1.  **Resolución Fina (Semanas -2 a 35):** El núcleo del curso se modela con granularidad semanal para detectar patrones precisos de riesgo.
2.  **Agregación de Colas (`Buckets`):**
    *   `w_prev`: Acumula toda la actividad histórica previa a la semana -2 (fase de matrícula y exploración temprana).
    *   `w_post`: Agrupa la actividad remanente tras la semana 35 (cierres administrativos y recuperaciones tardías).

---

### **Justificación Técnica: Bucketing Logarítmico**

Tras la generación de la matriz temporal, se observa que la distribución de la actividad sigue una **Ley de Potencia** (Power Law): la mayoría de estudiantes genera pocos clics semanales, mientras que una minoría ("power users") genera miles.

Esta dispersión extrema dificulta la convergencia de las redes neuronales, cuyos gradientes son sensibles a las magnitudes de entrada. Para mitigar este problema sin eliminar la información de los valores altos, se aplica una transformación no lineal:

$$ x' = \log(1 + x) $$

**Resultados de la Transformación:**
*   **Compresión de Rango:** La escala original [0, ~7000] se reduce a un espacio denso [0, ~9.0], mucho más amigable para el entrenamiento.
*   **Normalización de Distribución:** Se suaviza el sesgo (skewness) de la distribución, acercándola a una normalidad que favorece el aprendizaje de los pesos en las capas de atención del Transformer.
