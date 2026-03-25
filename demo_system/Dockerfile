FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120"]
