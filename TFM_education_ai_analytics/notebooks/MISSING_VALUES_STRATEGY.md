# Análisis y Tratamiento de Valores Nulos

En el preprocesamiento inicial (`dataset.py`) hemos convertido los valores perdidos codificados (como '?') a formato `NaN`. A continuación, detallamos dónde se encuentran estos nulos y la estrategia de imputación que seguiremos para la generación de features:

### 1. Interacciones (`interactions_processed.csv`)
*   **Campos Afectados**: `week_from`, `week_to`.
*   **Cantidad**: ~18 millones (la inmensa mayoría).
*   **Causa**: Estos campos indican la semana "sugerida" para estudiar un material. Muchos recursos (foros, página de inicio, glosarios) son transversales y no tienen una semana específica asignada.
*   **Tratamiento**:
    *   **No eliminar**: Perderíamos casi todo el dataset de interacciones.
    *   **Imputación**: No es necesaria para nuestro modelo temporal. Utilizaremos el campo `date` (fecha real del clic) para ubicar la interacción en la semana lectiva correspondiente (`week = date // 7`). Los campos `week_from/to` se descartarán para el modelado dinámico.

### 2. Estudiantes (`students_processed.csv`)
*   **Campo Afectado**: `date_unregistration`
*   **Cantidad**: 23,677 registros.
*   **Causa**: El valor es nulo si el estudiante **NO abandonó** el curso.
*   **Tratamiento**:
    *   **Imputación Lógica**: Sustituiremos los `NaN` por un valor fuera de rango positivo (ej. `999`) o la longitud máxima del curso. Esto indicará explícitamente al modelo que el evento de abandono no ocurrió durante el periodo lectivo estándar.

### 3. Evaluaciones (`assessments_processed.csv`)
*   **Campo Afectado**: `date` (Deadline).
*   **Cantidad**: 3,038 registros.
*   **Causa**: Son generalmente exámenes finales o evaluaciones continuas sin una fecha de corte estricta en la base de datos original.
*   **Tratamiento**:
    *   **Imputación por la media/moda del curso**: Si es crítico, se asignará la fecha del último día del curso (`code_presentation`).
    *   **Cálculo de Retraso**: Si no hay fecha límite, asumiremos `retraso = 0`.
