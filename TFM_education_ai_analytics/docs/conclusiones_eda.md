# Conclusiones del Análisis Exploratorio de Datos (EDA)
**Proyecto:** TFM Education AI Analytics
**Dataset:** Open University Learning Analytics Dataset (OULAD)

## 1. Resumen Ejecutivo
El análisis del dataset OULAD ha revelado patrones de comportamiento críticos que desmitifican la idea de que el "abandono" es un evento repentino. Hemos identificado que el fracaso es un **proceso gradual** con señales de alerta visibles incluso **antes del inicio del curso**. 

La arquitectura de IA propuesta (Transformer) es viable y necesaria para capturar la naturaleza secuencial y contextual de estos patrones.

---

## 2. Hallazgos Principales (The "Golden Nuggets")

### A. La "Purga" del Día 0 y la Alerta Temprana
*   **El Hallazgo:** Una gran parte de los abandonos (aprox. 40%) ocurre en las primeras semanas o incluso antes de empezar.
*   **La Causa:** Falta de compromiso inicial y miedo escénico.
*   **El Predictor:** 
    *   Los estudiantes que **NO entran al campus antes del Día 0** tienen un riesgo de abandono del **50%**.
    *   Los que sí "cotillean" antes de empezar reducen su riesgo al **25%**.
*   **Acción:** El modelo debe incluir `clics_pre_curso` y `dias_anticipacion_registro` como variables estáticas críticas.

### B. Calidad vs. Cantidad (Perfil de Navegación)
*   **El Hallazgo:** No se trata solo de cuántos clics haces, sino **dónde** los haces.
*   **El Patrón:**
    *   **Éxito (Distinction/Pass):** Uso intensivo del **Foro (`forumng`)**. El aprendizaje social correlaciona con la matrícula de honor.
    *   **Riesgo (Withdrawn/Fail):** Uso pasivo del **Temario (`oucontent`)** o nula interacción.
*   **Acción:** Feature Engineering debe desglosar los clics por `activity_type`, no solo contar totales.

### C. La Brecha Socioeconómica (Fairness)
*   **El Hallazgo:** El código postal predice el éxito. Los estudiantes de zonas desfavorecidas (`imd_band` 0-10%) tienen casi el doble de tasa de abandono que los de zonas ricas.
*   **Acción:** Es imperativo incluir `imd_band` en el modelo para que detecte este riesgo contextual y permita intervenciones de equidad (apoyo extra) en lugar de penalización algorítmica.

### D. Dinámica Temporal y la "Ley del Mínimo Esfuerzo"
*   **El Hallazgo:** Los estudiantes son **reactivos**. La actividad solo se dispara ante la inminencia de un *deadline* (evaluación).
*   **El Ritmo Diario:** Hemos confirmado que **todos los grupos** (Éxito y Fracaso) comparten el mismo patrón diario: alta carga Jueves/Viernes y descanso el fin de semana. No hay diferencias predictivas en "quién estudia el domingo".
*   **Acción:** La agregación de datos para la Red Neuronal debe ser **SEMANAL**. La granularidad diaria tiene demasiado "ruido" y no aporta valor diferencial.

---

## 3. Justificación de la Arquitectura (Transformer)

Los hallazgos validan el uso de un modelo híbrido:

1.  **Componente Secuencial (Encoder):** Necesario para entender la "historia" del alumno. Un alumno que pasa de "Mucho Foro" a "Nada" es más peligroso que uno que siempre tuvo "Poco Foro". Solo una RNN/Transformer capta este cambio de tendencia.
2.  **Componente Estático (Contexto):** Necesario para inyectar el riesgo base del **Día 0** (Demografía, Historial académico, Comportamiento pre-curso).

## 4. Próximos Pasos: Feature Engineering

Basado en esto, el plan de ingeniería de características es:

1.  **Features Estáticas (Contexto):** `imd_band`, `age_band`, `num_of_prev_attempts`, `pre_course_activity_flag`, `registration_lag_days`.
2.  **Features Dinámicas (Secuencia Semanal):**
    *   `clicks_forum`, `clicks_content`, `clicks_quiz` (separados).
    *   `total_clicks`.
    *   `assessments_submitted_flag` (para cruzar con deadlines).
3.  **Target:** `final_result` (Multiclase) o `is_withdrawn` (Binaria, para alerta super temprana).
