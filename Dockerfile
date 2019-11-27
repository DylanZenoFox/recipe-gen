FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ARG PYTHON_VERSION=3.7

COPY scripts/* /bin/
COPY requirements.txt /tmp/

EXPOSE 8888

RUN apt-get -y update \
    && apt-get install -y \
    software-properties-common \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    && add-apt-repository ppa:deadsnakes/ppa
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p opt/conda \
    && rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
    /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.2.0=py3.7_cuda10.0.130_cudnn7.6.2_0" \
    "torchvision=0.4.0=py37_cu100" \
 && conda clean -ya

RUN python3.7 -m pip install -r /tmp/requirements.txt
RUN python3.7 -m ipykernel install
RUN python3.7 -m spacy download en_core_web_sm
RUN jupyter contrib nbextension install --system
RUN jupyter nbextension enable --system codefolding/main
RUN jupyter nbextension enable --system scroll_down/main
RUN jupyter nbextension enable --system collapsible_headings/main
RUN jupyter nbextension enable --system execute_time/ExecuteTime
RUN jupyter nbextension enable --system init_cell/main
WORKDIR /code
CMD /bin/bash

