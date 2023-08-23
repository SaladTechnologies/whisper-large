FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ARG MODEL=openai/whisper-large-v2
ARG BRANCH=main
ENV MODEL_DIR=/models
ENV MODEL=${MODEL}
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y && apt-get upgrade -y && apt-get install --no-install-recommends --no-install-suggests -y \
  git \
  wget \
  libgl1 \
  libglib2.0-0 \
  build-essential \
  ffmpeg

COPY ./model/ ${MODEL_DIR}/${MODEL}

# We need the latest pip
RUN pip install --upgrade --no-cache-dir pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY web.py .
COPY model.py .
COPY Recording.wav .


CMD ["uvicorn", "web:app", "--port", "1111", "--host", "::"]