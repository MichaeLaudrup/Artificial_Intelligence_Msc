
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/TFM_education_ai_analytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    make \
    python3-venv \
    socat \
    wget \
    && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip


RUN git config --global user.email "maikelaudrupluisgonzalez@gmail.com" && \
    git config --global user.name "michaelaudrup"

COPY . /workspace/TFM_education_ai_analytics

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
