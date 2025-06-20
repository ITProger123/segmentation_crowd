FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
RUN mkdir /app/data
CMD ["python3", "main.py"]