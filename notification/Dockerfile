FROM python:3.11-slim

WORKDIR /app

COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY notification ./notification

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "notification.notification_service"]
