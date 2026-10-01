# CrowdSentry API + realtime processor image.
#
# One container runs both the HTTP API and the per-camera vision workers, so it must
# be deployed as exactly ONE process/replica (uvicorn --workers 1). To scale the API
# horizontally, run extra replicas with REALTIME_ENABLED=false and keep a single
# realtime-enabled instance -- see README "Production deployment".

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# opencv-python wheels need libGL and glib even though nothing here opens a window.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so source changes do not bust the (expensive) install layer.
# CPU-only torch goes in ahead of the rest: the default PyPI wheel on Linux drags in
# several GB of CUDA libraries this image never uses. A GPU variant would drop this
# line and set DETECTION_DEVICE=cuda.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY ml_artifacts ./ml_artifacts
COPY docker-entrypoint.sh .

# Bake the detection weights into the image: a bare "yolov8n.pt" is otherwise
# downloaded on first inference, which fails on egress-restricted networks and adds
# seconds to a cold start.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
 && useradd --system --uid 10001 crowdsentry \
 && chown -R crowdsentry /app

USER crowdsentry

EXPOSE 8000

ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
