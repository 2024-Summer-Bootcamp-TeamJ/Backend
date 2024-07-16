FROM python:slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
WORKDIR /app
