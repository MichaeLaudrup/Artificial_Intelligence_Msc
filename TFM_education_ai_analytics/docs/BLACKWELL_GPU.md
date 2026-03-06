# TensorFlow en RTX 5080 / Blackwell

## Ruta rápida

La wheel custom de TensorFlow no se instala automáticamente al levantar Docker.

Si quieres ir directo al grano:

```bash
make build_tf_blackwell_wheel
make install_tf_blackwell_wheel
make check_tf_blackwell
```

Interpretación rápida:

- si el último comando termina bien, la build de TensorFlow ya está lista para usar GPU en una comprobación sin datos
- si falla en modo estricto, la build instalada sigue sin ser compatible
- si no quieres reconstruir TensorFlow, el contenedor normal sigue pudiendo ejecutar en CPU

## Resumen

Este repositorio tuvo el problema típico de Blackwell con TensorFlow: la GPU se detectaba, pero el entrenamiento fallaba con errores de PTX al llegar a `model.fit()`.

El estado actual ya no es ese. La ruta GPU está resuelta y documentada para que otra persona no tenga que repetir la misma arqueología.

Punto clave:

- La solución real no fue un ajuste de flags, sino usar una build de TensorFlow compatible con Blackwell.
- Los workarounds residuales de XLA que se probaron durante la investigación ya se eliminaron del flujo normal.
- El repositorio conserva una protección útil: si alguien intenta usar una build incompatible en una GPU Blackwell, el runtime puede hacer fallback a CPU o fallar en seco si se pide modo estricto.

## Qué pasaba exactamente

Con wheels no preparadas para `sm_120`, TensorFlow llegaba a detectar la GPU pero terminaba rompiendo en entrenamiento con errores como este:

```text
LLVM ERROR: PTX version 8.5 does not support target 'sm_120'
```

Eso era especialmente confuso porque:

- `tf.config.list_physical_devices('GPU')` devolvía una GPU válida.
- Algunas operaciones pequeñas podían ejecutarse.
- El fallo aparecía tarde, normalmente en compilación o durante `model.fit()`.

## Qué hicimos

### 1. Aislar el problema real

Se descartó que el problema fuera del proyecto, del modelo o de Keras. El bloqueo era de compatibilidad entre:

- arquitectura Blackwell / `sm_120`
- versión efectiva de CUDA/PTX de la build de TensorFlow cargada
- toolchain usada para compilar la wheel

### 2. Añadir protección en el runtime del proyecto

El runtime central en [educational_ai_analytics/tf_runtime.py](/workspace/TFM_education_ai_analytics/educational_ai_analytics/tf_runtime.py) hace ahora dos cosas importantes:

- resuelve el dispositivo efectivo de ejecución
- detecta combinaciones GPU Blackwell + TensorFlow/CUDA incompatibles antes de llegar al entrenamiento

Comportamiento:

- si `EXECUTION_DEVICE=cpu`, oculta la GPU y ejecuta en CPU
- si `EXECUTION_DEVICE=gpu` pero la build es incompatible, hace fallback a CPU
- si además se define `TFM_GPU_STRICT=1`, falla de forma explícita en vez de degradar silenciosamente

Eso evita que otra persona pierda horas viendo una GPU “detectada” que en realidad no puede entrenar.

### 3. Construir una wheel compatible con Blackwell

La solución definitiva fue preparar una build propia de TensorFlow para esta familia de GPUs.

El repositorio incluye la ruta de build en:

- [scripts/build_tf220_wheel.sh](/workspace/TFM_education_ai_analytics/scripts/build_tf220_wheel.sh)
- [scripts/install_tf220_wheel.sh](/workspace/TFM_education_ai_analytics/scripts/install_tf220_wheel.sh)
- [Dockerfile.tf-builder](/workspace/TFM_education_ai_analytics/Dockerfile.tf-builder)

La estrategia usada fue:

- TensorFlow 2.22.x
- Clang 20 o superior
- CUDA hermética moderna
- cuDNN hermético moderno
- `TF_CUDA_COMPUTE_CAPABILITIES=12.0`

El script también incorpora parches prácticos para problemas de build detectados durante el proceso, por ejemplo el mapping de PTX en Bazel y el layout de `cuda_nvvm`.

## Estado final del repositorio

### Lo que sí se mantiene

- El chequeo de compatibilidad de Blackwell en el runtime.
- `TFM_GPU_STRICT=1` como forma de verificar compatibilidad real.
- Los scripts para construir e instalar la wheel custom.
- La separación de dispositivo por modelo mediante `execution_device` en cada `hyperparams.py`, con override opcional vía `EXECUTION_DEVICE`.

### Lo que se eliminó para no confundir

- `TF_XLA_FLAGS` en los entrypoints de entrenamiento.
- `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` en los entrypoints de entrenamiento.
- La idea de que el problema se resolvía “tocando flags”.
- La configuración global única de CPU/GPU en `config.py`.

La conclusión importante para quien llegue nuevo es simple: si TensorFlow no está bien construido para Blackwell, no hay combinación razonable de flags que convierta esa build en estable para entrenamiento.

## Cómo validar una máquina nueva

### Opción 1. Verificar que la build instalada ya sirve sin depender de datos del proyecto

```bash
make check_tf_blackwell
```

Esto ejecuta una validación mínima de runtime:

- imprime versión y `build_info` de TensorFlow
- lista las GPUs visibles
- fuerza el chequeo estricto de compatibilidad del runtime del proyecto
- ejecuta una operación pequeña en GPU sin necesitar dataset

Interpretación:

- si termina bien, la ruta GPU base está bien
- si falla con `TFM_GPU_STRICT=1`, la build no es válida para esa GPU
- si sin modo estricto hace fallback a CPU, el runtime te está protegiendo de una build incompatible

### Opción 2. Construir la wheel desde este repo

```bash
make build_tf_blackwell_wheel
make install_tf_blackwell_wheel
```

Después repite:

```bash
make check_tf_blackwell
```

### Opción 3. Validar entrenamiento real cuando ya tengas datos

Si además quieres validar el proyecto completo, entonces sí:

```bash
TFM_GPU_STRICT=1 EXECUTION_DEVICE=gpu make train_transformer
```

Ese paso ya depende de tener los datos del proyecto preparados.

## Comandos útiles

Entrenar explícitamente en CPU:

```bash
EXECUTION_DEVICE=cpu make train_transformer
```

Entrenar explícitamente en GPU y exigir compatibilidad real:

```bash
TFM_GPU_STRICT=1 EXECUTION_DEVICE=gpu make train_transformer
```

Chequeo rápido de TensorFlow sin datos:

```bash
make check_tf_blackwell
```

Benchmark de AE, sólo si ya tienes los datos del proyecto:

```bash
make bench_tf_blackwell
```

## Notas para mantenedores

- Si actualizas TensorFlow o el toolchain, vuelve a validar entrenamiento real, no sólo detección de GPU.
- Si reaparece un error de PTX, revisa primero la build efectiva de TensorFlow antes de tocar modelos, callbacks o flags de entorno.
- Si documentas otro hardware Blackwell, reutiliza esta guía pero evita vender como solución lo que sólo fue una etapa intermedia de depuración.