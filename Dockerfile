FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update --fix-missing -qq && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      git-lfs \
      openssh-client \
      openssh-server \
      locales \
      language-pack-en \
      netcat \
      sudo \
      stow \
      vim \
      wget \
      zsh && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

ARG USER=tianshu2020
ARG UID=1029
ARG GID=1029
RUN groupadd --gid ${GID} ${USER} && \
    useradd --shell /bin/zsh --uid ${UID} --gid ${GID} --create-home ${USER} && \
    echo ${USER}" ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${UID}:${GID}
SHELL ["/bin/zsh", "-c"]
WORKDIR /home/${USER}
RUN cd $HOME && \
    git clone --recurse-submodules https://github.com/tshu-w/dotfiles.git && \
    cd dotfiles && make link && exec zsh -i

ARG CONDA_ENV=env
ENV PATH=/home/${USER}/.local/share/conda/bin:${PATH}
RUN --mount=type=cache,uid=${UID},gid=${GID},target=/home/${USER}/.cache \
    curl -o miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash miniconda3.sh -b -p /home/${USER}/.local/share/conda && \
    rm miniconda3.sh && \
    conda update -n base -c defaults conda && \
    conda create -n ${CONDA_ENV} python

ENV PATH=/home/${USER}/.local/share/conda/envs/${CONDA_ENV}/bin:${PATH}
ENV DET_PYTHON_EXECUTABLE=/home/${USER}/.local/share/conda/bin/python3
COPY environment.yaml *requirements.txt ./
RUN --mount=type=cache,uid=${UID},gid=${GID},target=/home/${USER}/.cache \
    source activate $CONDA_ENV && \
    ([ ! -f environment.yaml ] || conda env update -n ${CONDA_ENV} -f environment.yaml) && \
    ([ ! -f requirements.txt ] || python -m pip install -r requirements.txt) && \
    rm requirements.txt environment.yaml

ARG DET_VERSION=0.19.10
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN --mount=type=cache,uid=${UID},gid=${GID},target=/home/${USER}/.cache python -m pip install --upgrade pip && \
    python -m pip install determined==${DET_VERSION} jupyterlab jupyter-archive jupyterlab-server
