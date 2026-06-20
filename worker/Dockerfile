FROM python:3.11-slim

WORKDIR /app

COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY worker ./worker
COPY inference/client.py ./inference/client.py
COPY inference/__init__.py ./inference/__init__.py

ENV PYTHONUNBUFFERED=1

# No HTTP port — this is a Pub/Sub pull subscriber, not a web service.
CMD ["python", "-m", "worker.image_processor"]
