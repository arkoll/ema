# 1. Test setup:
# docker run -it --rm --gpus all tensorflow/tensorflow:2.8.0-gpu nvidia-smi
#
# 2. Start training:
# docker build -f  agents/director/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh xvfb_run.sh python3 agents/director/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM tensorflow/tensorflow:2.9.0rc1-gpu
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
  apt ffmpeg git python3-pip vim wget unrar \
  && apt-get clean

# Envs
RUN pip3 install --no-cache-dir crafter
RUN pip3 install --no-cache-dir robodesk
RUN pip3 install --no-cache-dir dm_control
RUN pip3 install --no-cache-dir matplotlib
COPY scripts scripts
RUN sh scripts/install-atari.sh

ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

RUN apt-cache search mesa
RUN apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6

# Agent
RUN pip3 install --no-cache-dir dm-sonnet
RUN pip3 install --no-cache-dir tensorflow_probability==0.16.0
ENV TF_FUNCTION_JIT_COMPILE_DEFAULT 1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Embodied
RUN pip3 install --no-cache-dir numpy cloudpickle ruamel.yaml rich wandb
RUN pip3 install --upgrade requests
ENV WANDB_DIR /logdir
WORKDIR /embodied
