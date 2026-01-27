# Justificación de Valores Nulos (TFM)

Se ha decidido **no imputar ni eliminar** los valores nulos en el dataset OULAD, ya que representan información estructural del comportamiento estudiantil.

## Resumen de Decisiones

*   **`date_unregistration` (Estudiantes):** Un nulo indica que el alumno **no abandonó**. Imputarlo eliminaría la variable objetivo clave para predecir el fracaso escolar.
*   **`imd_band` (Estudiantes):** La falta de datos socioeconómicos se tratará como una categoría "Desconocida", evitando inventar perfiles sociales.
*   **`score` (Evaluaciones):** Representan el 0.1% de los datos; su ausencia es insignificante para el análisis global.
*   **`week_from / week_to` (Interacciones):** Se ignoran por completo (87% nulos). Usaremos la fecha real del clic (`date`) para la serie temporal.

## Conclusión
La limpieza tradicional de nulos provocaría **pérdida de información**. Al mantener los nulos con significado, permitimos que el **Transformer** aprenda de la realidad de los eventos, evitando sesgos artificiales por imputación.

## Hallazgos del Análisis Temporal

*   **Brecha de Actividad**: Se observa una clara separación entre estudiantes excelentes y en riesgo durante el ecuador del curso (Semanas 15-25). Los alumnos de alto rendimiento son proactivos, mientras que los de bajo rendimiento son reactivos a las entregas.
*   **Limitación de Hitos Globales**: Los picos de actividad no siempre coinciden con los *deadlines* globales de la universidad, lo que sugiere que para la IA será vital capturar patrones específicos de cada módulo.
*   **Señal vs Ruido**: La agregación semanal es fundamental para eliminar la volatilidad diaria y permitir que el modelo capture la trayectoria real del estudiante.
