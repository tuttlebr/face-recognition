# DOCKER_BUILDKIT=1 docker build -t 192.168.50.13:30256/tensorflow:20.08-tf2-py3 -f Dockerfile .

FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        wget \
        cmake \
        libx11-dev \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install progressbar2 fastapi uvicorn[standard] gunicorn pandas ipywidgets tensorflow-addons matplotlib

RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

#Install dlib
WORKDIR /

RUN git clone -b 'v19.21' --single-branch https://github.com/davisking/dlib.git dlib

WORKDIR /dlib

RUN mkdir -p /dlib/build

RUN cmake -H/dlib -B/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1

RUN cmake --build /dlib/build -- -j 10

RUN cd /dlib; python3 /dlib/setup.py install

# Kubeflow Requirements
ENV NB_PREFIX /

CMD ["sh","-c", "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
