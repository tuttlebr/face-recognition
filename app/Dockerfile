ARG NV_TENSORFLOW_BUILD=20.08-tf2-py3
FROM nvcr.io/nvidia/tensorflow:${NV_TENSORFLOW_BUILD}
COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
ENTRYPOINT /etc/bash.bashrc