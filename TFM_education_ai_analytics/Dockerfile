# Utilizamos la imagen oficial y optimizada de NVIDIA NGC para tener soporte nativo
# de las últimas arquitecturas de GPU (como la de la RTX 5080)
FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

# Evitar que python genere .pyc files en el contenedor
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Instalamos dependencias del sistema necesarias para desarrollo y Jupyter
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    make \
    python3-venv \
    socat \
    wget \
    jupyter \
    && rm -rf /var/lib/apt/lists/*

# Configuración global de Git
RUN git config --global user.email "maikelaudrupluisgonzalez@gmail.com" && \
    git config --global user.name "michaelaudrup"

# Copiamos todo el repositorio padre (el contexto de build es ..)
COPY . .

# Entramos en la subcarpeta del proyecto
WORKDIR /workspace/TFM_education_ai_analytics

# Instalamos dependencias de Python en un venv fuera de /workspace para que
# no se rompa cuando el proyecto se monte como bind volume en otro PC.
RUN python -m venv --system-site-packages /opt/tfm-venv

# Activar el entorno virtual automáticamente en el contenedor.
ENV PATH="/opt/tfm-venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# Registramos el kernel de Jupyter con el nombre del proyecto.
RUN python -m ipykernel install --user --name educational_ai_analytics --display-name "Python (TFM Analytics)"

# Configuramos el entrypoint (ahora está en el directorio actual)
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Comando por defecto para mantener el contenedor vivo si se ejecuta solo
CMD ["tail", "-f", "/dev/null"]
