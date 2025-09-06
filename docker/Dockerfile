# syntax=docker/dockerfile:1

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# Set INSTALL_GPU_EXTRAS=true at build time to install optional GPU features
ARG INSTALL_GPU_EXTRAS=false

# Ensure Python is available when using non-Python base images
RUN if ! command -v python >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
      ln -sf python3 /usr/bin/python && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
COPY requirements-core.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-core.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e . \
    && if [ "$INSTALL_GPU_EXTRAS" = "true" ]; then pip install --no-cache-dir .[gpu]; fi \
    && chmod +x docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python"]
