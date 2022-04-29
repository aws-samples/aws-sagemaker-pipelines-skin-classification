ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++-5 \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev vim feh wget xterm

# Install Miniconda
RUN wget -O /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install tqdm sklearn opencv-python h5py Pillow scikit-image boto3 sagemaker scikit-learn opencv-contrib-python mxnet
RUN pip install imblearn
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]