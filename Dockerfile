FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt && rm -rf /tmp/*

COPY . .
COPY download_models.sh .
RUN chmod +x download_models.sh

RUN mkdir -p models uploads outputs

EXPOSE 8000

CMD ["/bin/bash", "-c", "./download_models.sh && gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 120 app.main:app"]
