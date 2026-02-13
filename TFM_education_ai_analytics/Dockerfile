FROM tensorflow/tensorflow:latest-gpu-jupyter

# Evitar que python genere .pyc files en el contenedor
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Instalamos dependencias del sistema necesarias para desarrollo y VS Code Server
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    make \
    socat \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Configuración global de Git
RUN git config --global user.email "maikelaudrupluisgonzalez@gmail.com" && \
    git config --global user.name "michaelaudrup"

# Copiamos todo el repositorio padre (el contexto de build es ..)
COPY . .

# Entramos en la subcarpeta del proyecto
WORKDIR /workspace/TFM_education_ai_analytics

# Instalamos dependencias de Python
# Nota: tensorflow ya viene instalado en la imagen base, pip install . instalará las demás
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# Registramos el kernel de Jupyter con el nombre del proyecto
RUN python -m ipykernel install --user --name educational_ai_analytics --display-name "Python (TFM Analytics)"

# Configuramos el entrypoint (ahora está en el directorio actual)
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Comando por defecto para mantener el contenedor vivo si se ejecuta solo
CMD ["tail", "-f", "/dev/null"]
