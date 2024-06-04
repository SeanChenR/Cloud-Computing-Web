FROM --platform=linux/amd64 python:3

RUN mkdir /app
WORKDIR /app

COPY final_web.py .
COPY translator.py .
COPY vision.py .
COPY conversation.py .
COPY blob_list.py .
COPY storage_upload.py .
COPY TWSC_embedding.py .
COPY requirements.txt .
COPY config.ini .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "final_web.py", "--server.port=8501", "--server.address=0.0.0.0"]
