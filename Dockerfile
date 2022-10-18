FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
# using pytorch:1.12.0-cuda11.3-cudnn8-devel results in training being 2x slower for some weird reason

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update \
 && apt-get install ffmpeg wget git -y

RUN pip install \
        opencv-python \
        pillow \
        matplotlib \
        scikit-learn \
        scipy \
        tqdm \
        pandas \
        ffmpeg-python \
        ftfy \
        regex \
        imgaug

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY cuda_ops /tmp

RUN cd /tmp \
 && TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6" python setup.py install \
 && rm -rf *
