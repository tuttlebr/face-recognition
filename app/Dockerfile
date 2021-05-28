ARG NV_TENSORFLOW_BUILD=20.08-tf2-py3

FROM nvcr.io/nvidia/tensorflow:${NV_TENSORFLOW_BUILD}

ARG CORE_COUNT=4
ENV DEBIAN_FRONTEND=noninteractive
ENV AUTOGRAPH_VERBOSITY=0

COPY requirements.txt /requirements.txt

COPY entrypoint.sh  /usr/local/bin/nvidia_entrypoint.sh

WORKDIR /models

ADD http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 \
    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/mmod_human_face_detector.dat.bz2 \
    /models/

WORKDIR /

RUN chmod a+rwx /usr/local/bin/nvidia_entrypoint.sh \
    && echo -e " => Installing runtime packages" \
    && apt-get update \
    && apt-get install -yq --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    liblapack-dev \
    libopenblas-dev \
    libx11-dev \
    wget \
    && apt-get autoremove -yq \
    && apt-get clean -yq \
    && apt-get -yq autoclean \
    && echo " => Decompressing DLIB models" \
    && bzip2 -d /models/dlib_face_recognition_resnet_model_v1.dat.bz2 \
    && bzip2 -d /models/shape_predictor_5_face_landmarks.dat.bz2 \
    && bzip2 -d /models/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d /models/mmod_human_face_detector.dat.bz2 \
    && echo -e " => Installing python dependencies" \
    && pip install -r /requirements.txt \
    && rm /requirements.txt \
    && echo -e " => Building DLIB with CUDA support" \
    && git clone -b 'v19.21' --single-branch https://github.com/davisking/dlib.git \
    && mkdir -p /dlib/build \
    && cmake -H/dlib -B/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 \
    && cmake --build /dlib/build -- -j ${CORE_COUNT} \
    && cd /dlib; python3 /dlib/setup.py install

WORKDIR /photos

WORKDIR /app
COPY . .

CMD ["python3", "/app/00_find_faces.py"]