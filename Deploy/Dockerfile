FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_23.5.2-0-Linux-x86_64.sh

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

COPY . /workspace
WORKDIR /workspace

CMD streamlit run app.py