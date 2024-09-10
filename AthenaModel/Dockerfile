# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:10.1-base
FROM python:3.9.16
#FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

EXPOSE 8000
USER root



# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libsndfile1
RUN apt-get install -y runit runit-systemd
RUN apt-get -y install git
# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt



WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

RUN pip install git+https://github.com/elliottzheng/batch-face.git@master
#USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["gunicorn","-w","1","--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "inference:app"]
#CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]