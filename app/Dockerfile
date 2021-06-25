ARG NV_TENSORFLOW_BUILD=20.08-tf2-py3
FROM nvcr.io/nvidia/tensorflow:${NV_TENSORFLOW_BUILD}

ARG USER=brandon
ARG CORE_COUNT=4

ENV DEBIAN_FRONTEND=noninteractive
ENV AUTOGRAPH_VERBOSITY=0
ENV HOME=/home/${USER}

COPY entrypoint.sh  /usr/local/bin/nvidia_entrypoint.sh

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
    && apt-get -yq autoclean

WORKDIR ${HOME}

COPY requirements.txt requirements.txt

ADD http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 \
    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/mmod_human_face_detector.dat.bz2 \
    ${HOME}/models/

RUN echo " => Decompressing DLIB models" \
    && bzip2 -d ${HOME}/models/dlib_face_recognition_resnet_model_v1.dat.bz2 \
    && bzip2 -d ${HOME}/models/shape_predictor_5_face_landmarks.dat.bz2 \
    && bzip2 -d ${HOME}/models/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d ${HOME}/models/mmod_human_face_detector.dat.bz2 \
    && echo -e " => Installing python dependencies" \
    && pip install -r ${HOME}/requirements.txt \
    && rm ${HOME}/requirements.txt \
    && echo -e " => Building DLIB with CUDA support" \
    && git clone -b 'v19.21' --single-branch https://github.com/davisking/dlib.git \
    && mkdir -p ${HOME}/dlib/build \
    && cmake -H${HOME}/dlib -B${HOME}/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 \
    && cmake --build ${HOME}/dlib/build -- -j ${CORE_COUNT} \
    && cd ${HOME}/dlib; python3 ${HOME}/dlib/setup.py install

COPY . .

RUN useradd -M -s /bin/bash -N -u 1000 ${USER} \
    && chown -R ${USER}:users /usr/local/bin \
    && chown -R ${USER}:users ${HOME}

USER ${USER}

CMD ["sh","-c", "python3 00_find_faces.py"]