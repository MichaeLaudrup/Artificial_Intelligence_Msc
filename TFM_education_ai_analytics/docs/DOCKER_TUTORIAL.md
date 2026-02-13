<<<<<<< HEAD
# 🐳 Tutorial Docker para el TFM — Guía para Dummies

## ¿Por qué Docker?

Tu proyecto de TFM usa **TensorFlow con GPU (CUDA)**. El problema es que CUDA solo funciona
nativamente en Linux, y tú trabajas en Windows. Docker soluciona esto creando un **contenedor
Linux** dentro de tu Windows que tiene acceso directo a tu GPU (RTX 3070).

```
┌─────────────────────────────────────────────┐
│  TU WINDOWS                                 │
│                                             │
│  ┌────────────────────────────────────────┐  │
│  │  CONTENEDOR LINUX (Ubuntu)            │  │
│  │                                       │  │
│  │  • Python 3.11                        │  │
│  │  • TensorFlow + CUDA                  │  │
│  │  • Tus dependencias (pandas, etc.)    │  │
│  │  • Tu código (montado como volumen)   │  │
│  │                                       │  │
│  │  🔗 Acceso a tu GPU RTX 3070         │  │
│  └────────────────────────────────────────┘  │
│                                             │
│  VS Code se conecta al contenedor          │
│  como si estuvieras en Linux               │
└─────────────────────────────────────────────┘
```

---

## 📁 Archivos involucrados (4 piezas del puzzle)

El sistema usa **4 archivos** que trabajan juntos. Piensa en ellos como capas:

| Archivo | ¿Qué hace? | Analogía |
|---------|------------|----------|
| `Dockerfile` | Define cómo construir la imagen | La **receta** de cocina |
| `docker-compose.yml` | Define cómo ejecutar el contenedor | Las **instrucciones de servir** |
| `.devcontainer/devcontainer.json` | Conecta VS Code al contenedor | El **puente** entre tu editor y el contenedor |
| `.dockerignore` | Dice qué archivos NO enviar a Docker | La **lista de exclusión** |

---

## 🔧 Archivo 1: `Dockerfile` — La Receta

```dockerfile
FROM tensorflow/tensorflow:latest-gpu-jupyter
```
**Línea 1 — La imagen base.** Es como decir "empieza con un Ubuntu que ya tiene TensorFlow,
CUDA, y Jupyter instalados". No necesitas instalar nada de eso manualmente.

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
```
**Variables de entorno.**
- `PYTHONDONTWRITEBYTECODE=1`: Evita que Python cree archivos `.pyc` (basura).
- `PYTHONUNBUFFERED=1`: Los prints de Python salen inmediatamente (útil para logs).

```dockerfile
WORKDIR /workspace
```
**Directorio de trabajo.** Todos los comandos siguientes se ejecutan desde `/workspace`.
Es como hacer `cd /workspace` al principio.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    make \
    socat \
    wget \
    && rm -rf /var/lib/apt/lists/*
```
**Paquetes del sistema.** Instalamos herramientas que necesita el contenedor:
- `ca-certificates`: Para conexiones HTTPS seguras.
- `curl` / `wget`: Para que VS Code pueda descargarse su servidor remoto dentro del contenedor.
- `git`: Para control de versiones.
- `make`: Para ejecutar el Makefile del proyecto.
- `socat`: Para que VS Code pueda hacer port-forwarding (túneles de red).
- `rm -rf /var/lib/apt/lists/*`: Limpia la caché de apt para que la imagen sea más pequeña.

```dockerfile
COPY pyproject.toml README.md LICENSE ./
COPY educational_ai_analytics ./educational_ai_analytics
```
**Copiamos archivos necesarios.** Para instalar el paquete Python, `flit` (nuestro build system)
necesita:
- `pyproject.toml`: Define el proyecto y sus dependencias.
- `README.md`: Lo usa flit para la descripción del paquete.
- `LICENSE`: Referenciado en pyproject.toml.
- `educational_ai_analytics/`: El código fuente del paquete.

> ⚠️ **No copiamos data, notebooks, models, etc.** porque esos se montan como volumen
> (ver docker-compose.yml). Solo copiamos lo mínimo para instalar dependencias.

```dockerfile
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .
```
**Instalamos las dependencias Python** definidas en `pyproject.toml` (pandas, numpy, scikit-learn,
etc.). TensorFlow NO está en la lista porque ya viene preinstalado en la imagen base.
- `--no-cache-dir`: No guarda caché de pip → imagen más pequeña.

```dockerfile
CMD ["tail", "-f", "/dev/null"]
```
**Comando por defecto.** Mantiene el contenedor vivo sin hacer nada. Es como un
"quédate encendido y espera instrucciones". VS Code luego se conectará y lo usará.

---

## 🎼 Archivo 2: `docker-compose.yml` — Las Instrucciones de Servir

```yaml
services:
  tfm-gpu:               # Nombre del servicio
    build: .              # Construye usando el Dockerfile de esta carpeta
    volumes:
      - .:/workspace      # ⭐ CLAVE: monta tu carpeta local dentro del contenedor
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]   # ⭐ Reserva tu GPU NVIDIA
    command: tail -f /dev/null      # Mantener vivo
    restart: unless-stopped         # Se reinicia si se cae
    ports:
      - "8888:8888"                 # Puerto de Jupyter
```

### Lo más importante aquí:

**`volumes: - .:/workspace`** — Esto es la MAGIA. Monta tu carpeta del proyecto
de Windows directamente dentro del contenedor en `/workspace`. Significa que:
- Los archivos que edites en Windows **aparecen instantáneamente** en el contenedor.
- Los archivos que cree el contenedor **aparecen en tu Windows**.
- No necesitas copiar nada manualmente. Es como un espejo.

```
TU WINDOWS:                              CONTENEDOR LINUX:
C:\Users\maike\...\TFM_education...  ←→  /workspace
├── data/                            ←→  ├── data/
├── notebooks/                       ←→  ├── notebooks/
├── educational_ai_analytics/        ←→  ├── educational_ai_analytics/
└── ...                              ←→  └── ...
```

**`deploy.resources.reservations.devices`** — Le dice a Docker: "este contenedor necesita
una GPU NVIDIA". Docker Desktop + WSL2 + NVIDIA Container Toolkit se encargan de pasar
tu RTX 3070 al contenedor Linux.

---

## 🌉 Archivo 3: `.devcontainer/devcontainer.json` — El Puente VS Code ↔ Contenedor

```jsonc
{
  "name": "TFM GPU Container",
  "dockerComposeFile": "../docker-compose.yml",   // Usa el docker-compose
  "service": "tfm-gpu",                           // Se conecta a este servicio
  "workspaceFolder": "/workspace",                // Abre esta carpeta al conectar

  "customizations": {
    "vscode": {
      "extensions": [                              // Instala estas extensiones
        "ms-python.python",                        // dentro del contenedor
        "ms-toolsai.jupyter",
        "ms-azuretools.vscode-docker"
      ]
    }
  },

  "postCreateCommand": "cd /workspace && pip install --no-deps -e .",

  "remoteUser": "root"
}
```

### ¿Qué hace cada cosa?

- **`dockerComposeFile`**: Le dice a VS Code "usa docker-compose.yml para levantar el contenedor".
- **`service: "tfm-gpu"`**: "Conéctate al servicio llamado tfm-gpu".
- **`workspaceFolder`**: "Cuando abras el editor, ábrelo en /workspace".
- **`extensions`**: Instala Python, Jupyter y Docker extensions **dentro del contenedor**
  (separadas de las que tienes en Windows).
- **`postCreateCommand`**: Se ejecuta después de crear el contenedor. Instala tu paquete
  en modo "editable" (`-e .`) para que los cambios en tu código se reflejen sin reinstalar.
  `--no-deps` evita reinstalar dependencias que ya están en la imagen.
- **`remoteUser: "root"`**: Usa el usuario root dentro del contenedor (más simple para desarrollo).

---

## 🚫 Archivo 4: `.dockerignore` — La Lista de Exclusión

```
data
models
notebooks
.git
.venv
...
```

Cuando Docker construye la imagen, envía todos los archivos de tu carpeta al "Docker daemon"
como **contexto de build**. Sin `.dockerignore`, enviaría TODO (datos, modelos,
virtualenv, historial git...) lo cual haría el build lentísimo.

Este archivo dice: "no envíes estas carpetas, no las necesitas para construir la imagen".
No las necesitamos porque se montan como volumen después.

---

## 🔄 Flujo completo: ¿Qué pasa cuando haces "Reopen in Container"?

```
Tú pulsas "Reopen in Container"
         │
         ▼
    ┌─────────────┐
    │ 1. BUILD     │  Docker lee el Dockerfile y construye la imagen
    │              │  (instala paquetes, dependencias Python, etc.)
    │              │  Solo tarda la primera vez. Después usa caché.
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 2. CREATE    │  Docker crea el contenedor usando docker-compose.yml
    │              │  (monta volúmenes, reserva GPU, abre puertos)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 3. START     │  Docker arranca el contenedor
    │              │  (ejecuta "tail -f /dev/null" para mantenerlo vivo)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 4. CONNECT   │  VS Code instala su servidor remoto dentro del
    │              │  contenedor (necesita wget + socat)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 5. POST      │  Ejecuta postCreateCommand:
    │    CREATE     │  "pip install --no-deps -e ." → instala tu paquete
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 6. READY ✅  │  VS Code abre tu workspace dentro del contenedor
    │              │  ¡Ya estás en Linux con GPU!
    └─────────────┘
```

---

## 🛠️ Comandos útiles

| Qué quieres hacer | Comando |
|---|---|
| Ver contenedores activos | `docker ps` |
| Ver logs del contenedor | `docker logs tfm_education_ai_analytics-tfm-gpu-1` |
| Entrar al contenedor manualmente | `docker exec -it tfm_education_ai_analytics-tfm-gpu-1 bash` |
| Parar todo | `docker compose down` |
| Parar y borrar imagen | `docker compose down --rmi all` |
| Reconstruir desde cero | `docker compose build --no-cache` |
| Verificar GPU dentro del contenedor | `nvidia-smi` (desde la terminal del contenedor) |

---

## ❓ Problemas que tuvimos y por qué

| Error | Causa | Solución |
|-------|-------|----------|
| `README.md does not exist` | El Dockerfile solo copiaba `pyproject.toml`, pero `flit` necesita también README.md y LICENSE | Añadir `COPY README.md LICENSE ./` |
| `Error need wget to download server binary` | VS Code necesita descargar su servidor dentro del contenedor | Añadir `wget` al `apt-get install` |
| `subprocess terminated with return code 127` | VS Code necesita `socat` para port-forwarding | Añadir `socat` al `apt-get install` |
| `file:///root does not appear to be a Python project` | `postCreateCommand` se ejecutaba en `/root` en vez de `/workspace` | Cambiar a `cd /workspace && pip install ...` |
| Build muy lento | Sin `.dockerignore`, Docker enviaba data, .venv, .git, etc. | Crear `.dockerignore` |
| pip tardaba siglos | `tensorflow` estaba en `pyproject.toml` pero ya viene en la imagen base | Quitarlo de las dependencias |
=======
# 🐳 Tutorial Docker para el TFM — Guía para Dummies

## ¿Por qué Docker?

Tu proyecto de TFM usa **TensorFlow con GPU (CUDA)**. El problema es que CUDA solo funciona
nativamente en Linux, y tú trabajas en Windows. Docker soluciona esto creando un **contenedor
Linux** dentro de tu Windows que tiene acceso directo a tu GPU (RTX 3070).

```
┌─────────────────────────────────────────────┐
│  TU WINDOWS                                 │
│                                             │
│  ┌────────────────────────────────────────┐  │
│  │  CONTENEDOR LINUX (Ubuntu)            │  │
│  │                                       │  │
│  │  • Python 3.11                        │  │
│  │  • TensorFlow + CUDA                  │  │
│  │  • Tus dependencias (pandas, etc.)    │  │
│  │  • Tu código (montado como volumen)   │  │
│  │                                       │  │
│  │  🔗 Acceso a tu GPU RTX 3070         │  │
│  └────────────────────────────────────────┘  │
│                                             │
│  VS Code se conecta al contenedor          │
│  como si estuvieras en Linux               │
└─────────────────────────────────────────────┘
```

---

## 📁 Archivos involucrados (4 piezas del puzzle)

El sistema usa **4 archivos** que trabajan juntos. Piensa en ellos como capas:

| Archivo | ¿Qué hace? | Analogía |
|---------|------------|----------|
| `Dockerfile` | Define cómo construir la imagen | La **receta** de cocina |
| `docker-compose.yml` | Define cómo ejecutar el contenedor | Las **instrucciones de servir** |
| `.devcontainer/devcontainer.json` | Conecta VS Code al contenedor | El **puente** entre tu editor y el contenedor |
| `.dockerignore` | Dice qué archivos NO enviar a Docker | La **lista de exclusión** |

---

## 🔧 Archivo 1: `Dockerfile` — La Receta

```dockerfile
FROM tensorflow/tensorflow:latest-gpu-jupyter
```
**Línea 1 — La imagen base.** Es como decir "empieza con un Ubuntu que ya tiene TensorFlow,
CUDA, y Jupyter instalados". No necesitas instalar nada de eso manualmente.

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
```
**Variables de entorno.**
- `PYTHONDONTWRITEBYTECODE=1`: Evita que Python cree archivos `.pyc` (basura).
- `PYTHONUNBUFFERED=1`: Los prints de Python salen inmediatamente (útil para logs).

```dockerfile
WORKDIR /workspace
```
**Directorio de trabajo.** Todos los comandos siguientes se ejecutan desde `/workspace`.
Es como hacer `cd /workspace` al principio.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    make \
    socat \
    wget \
    && rm -rf /var/lib/apt/lists/*
```
**Paquetes del sistema.** Instalamos herramientas que necesita el contenedor:
- `ca-certificates`: Para conexiones HTTPS seguras.
- `curl` / `wget`: Para que VS Code pueda descargarse su servidor remoto dentro del contenedor.
- `git`: Para control de versiones.
- `make`: Para ejecutar el Makefile del proyecto.
- `socat`: Para que VS Code pueda hacer port-forwarding (túneles de red).
- `rm -rf /var/lib/apt/lists/*`: Limpia la caché de apt para que la imagen sea más pequeña.

```dockerfile
COPY pyproject.toml README.md LICENSE ./
COPY educational_ai_analytics ./educational_ai_analytics
```
**Copiamos archivos necesarios.** Para instalar el paquete Python, `flit` (nuestro build system)
necesita:
- `pyproject.toml`: Define el proyecto y sus dependencias.
- `README.md`: Lo usa flit para la descripción del paquete.
- `LICENSE`: Referenciado en pyproject.toml.
- `educational_ai_analytics/`: El código fuente del paquete.

> ⚠️ **No copiamos data, notebooks, models, etc.** porque esos se montan como volumen
> (ver docker-compose.yml). Solo copiamos lo mínimo para instalar dependencias.

```dockerfile
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .
```
**Instalamos las dependencias Python** definidas en `pyproject.toml` (pandas, numpy, scikit-learn,
etc.). TensorFlow NO está en la lista porque ya viene preinstalado en la imagen base.
- `--no-cache-dir`: No guarda caché de pip → imagen más pequeña.

```dockerfile
CMD ["tail", "-f", "/dev/null"]
```
**Comando por defecto.** Mantiene el contenedor vivo sin hacer nada. Es como un
"quédate encendido y espera instrucciones". VS Code luego se conectará y lo usará.

---

## 🎼 Archivo 2: `docker-compose.yml` — Las Instrucciones de Servir

```yaml
services:
  tfm-gpu:               # Nombre del servicio
    build: .              # Construye usando el Dockerfile de esta carpeta
    volumes:
      - .:/workspace      # ⭐ CLAVE: monta tu carpeta local dentro del contenedor
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]   # ⭐ Reserva tu GPU NVIDIA
    command: tail -f /dev/null      # Mantener vivo
    restart: unless-stopped         # Se reinicia si se cae
    ports:
      - "8888:8888"                 # Puerto de Jupyter
```

### Lo más importante aquí:

**`volumes: - .:/workspace`** — Esto es la MAGIA. Monta tu carpeta del proyecto
de Windows directamente dentro del contenedor en `/workspace`. Significa que:
- Los archivos que edites en Windows **aparecen instantáneamente** en el contenedor.
- Los archivos que cree el contenedor **aparecen en tu Windows**.
- No necesitas copiar nada manualmente. Es como un espejo.

```
TU WINDOWS:                              CONTENEDOR LINUX:
C:\Users\maike\...\TFM_education...  ←→  /workspace
├── data/                            ←→  ├── data/
├── notebooks/                       ←→  ├── notebooks/
├── educational_ai_analytics/        ←→  ├── educational_ai_analytics/
└── ...                              ←→  └── ...
```

**`deploy.resources.reservations.devices`** — Le dice a Docker: "este contenedor necesita
una GPU NVIDIA". Docker Desktop + WSL2 + NVIDIA Container Toolkit se encargan de pasar
tu RTX 3070 al contenedor Linux.

---

## 🌉 Archivo 3: `.devcontainer/devcontainer.json` — El Puente VS Code ↔ Contenedor

```jsonc
{
  "name": "TFM GPU Container",
  "dockerComposeFile": "../docker-compose.yml",   // Usa el docker-compose
  "service": "tfm-gpu",                           // Se conecta a este servicio
  "workspaceFolder": "/workspace",                // Abre esta carpeta al conectar

  "customizations": {
    "vscode": {
      "extensions": [                              // Instala estas extensiones
        "ms-python.python",                        // dentro del contenedor
        "ms-toolsai.jupyter",
        "ms-azuretools.vscode-docker"
      ]
    }
  },

  "postCreateCommand": "cd /workspace && pip install --no-deps -e .",

  "remoteUser": "root"
}
```

### ¿Qué hace cada cosa?

- **`dockerComposeFile`**: Le dice a VS Code "usa docker-compose.yml para levantar el contenedor".
- **`service: "tfm-gpu"`**: "Conéctate al servicio llamado tfm-gpu".
- **`workspaceFolder`**: "Cuando abras el editor, ábrelo en /workspace".
- **`extensions`**: Instala Python, Jupyter y Docker extensions **dentro del contenedor**
  (separadas de las que tienes en Windows).
- **`postCreateCommand`**: Se ejecuta después de crear el contenedor. Instala tu paquete
  en modo "editable" (`-e .`) para que los cambios en tu código se reflejen sin reinstalar.
  `--no-deps` evita reinstalar dependencias que ya están en la imagen.
- **`remoteUser: "root"`**: Usa el usuario root dentro del contenedor (más simple para desarrollo).

---

## 🚫 Archivo 4: `.dockerignore` — La Lista de Exclusión

```
data
models
notebooks
.git
.venv
...
```

Cuando Docker construye la imagen, envía todos los archivos de tu carpeta al "Docker daemon"
como **contexto de build**. Sin `.dockerignore`, enviaría TODO (datos, modelos,
virtualenv, historial git...) lo cual haría el build lentísimo.

Este archivo dice: "no envíes estas carpetas, no las necesitas para construir la imagen".
No las necesitamos porque se montan como volumen después.

---

## 🔄 Flujo completo: ¿Qué pasa cuando haces "Reopen in Container"?

```
Tú pulsas "Reopen in Container"
         │
         ▼
    ┌─────────────┐
    │ 1. BUILD     │  Docker lee el Dockerfile y construye la imagen
    │              │  (instala paquetes, dependencias Python, etc.)
    │              │  Solo tarda la primera vez. Después usa caché.
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 2. CREATE    │  Docker crea el contenedor usando docker-compose.yml
    │              │  (monta volúmenes, reserva GPU, abre puertos)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 3. START     │  Docker arranca el contenedor
    │              │  (ejecuta "tail -f /dev/null" para mantenerlo vivo)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 4. CONNECT   │  VS Code instala su servidor remoto dentro del
    │              │  contenedor (necesita wget + socat)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 5. POST      │  Ejecuta postCreateCommand:
    │    CREATE     │  "pip install --no-deps -e ." → instala tu paquete
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 6. READY ✅  │  VS Code abre tu workspace dentro del contenedor
    │              │  ¡Ya estás en Linux con GPU!
    └─────────────┘
```

---

## 🛠️ Comandos útiles

| Qué quieres hacer | Comando |
|---|---|
| Ver contenedores activos | `docker ps` |
| Ver logs del contenedor | `docker logs tfm_education_ai_analytics-tfm-gpu-1` |
| Entrar al contenedor manualmente | `docker exec -it tfm_education_ai_analytics-tfm-gpu-1 bash` |
| Parar todo | `docker compose down` |
| Parar y borrar imagen | `docker compose down --rmi all` |
| Reconstruir desde cero | `docker compose build --no-cache` |
| Verificar GPU dentro del contenedor | `nvidia-smi` (desde la terminal del contenedor) |

---

## ❓ Problemas que tuvimos y por qué

| Error | Causa | Solución |
|-------|-------|----------|
| `README.md does not exist` | El Dockerfile solo copiaba `pyproject.toml`, pero `flit` necesita también README.md y LICENSE | Añadir `COPY README.md LICENSE ./` |
| `Error need wget to download server binary` | VS Code necesita descargar su servidor dentro del contenedor | Añadir `wget` al `apt-get install` |
| `subprocess terminated with return code 127` | VS Code necesita `socat` para port-forwarding | Añadir `socat` al `apt-get install` |
| `file:///root does not appear to be a Python project` | `postCreateCommand` se ejecutaba en `/root` en vez de `/workspace` | Cambiar a `cd /workspace && pip install ...` |
| Build muy lento | Sin `.dockerignore`, Docker enviaba data, .venv, .git, etc. | Crear `.dockerignore` |
| pip tardaba siglos | `tensorflow` estaba en `pyproject.toml` pero ya viene en la imagen base | Quitarlo de las dependencias |
>>>>>>> c30dc9262eee8dc25b98d7ef8a910c40c00b5fda
