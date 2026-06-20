# Minimal CPU-only image — no CUDA/GPU layers, keeps the image small and
# the pod schedulable on cheap standard GKE node pools (no GPU quota
# needed), which matters for the "horizontally scalable / bursty traffic"
# requirement in the brief.
FROM python:3.11-slim

WORKDIR /app

COPY inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/model_server ./inference/model_server
COPY models ./models

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# 2 worker processes per pod; horizontal scaling for further capacity is
# handled by the HPA in k8s/inference-hpa.yaml, not by stacking more
# workers into a single pod (keeps per-pod memory predictable).
CMD ["uvicorn", "inference.model_server.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
