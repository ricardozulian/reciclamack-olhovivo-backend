FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN adduser --disabled-password --no-create-home --gecos "" appuser \
 && mkdir -p /app/app/data /app/app/uploads \
 && chown -R appuser:appuser /app

EXPOSE 8000

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
