# check the pytorch/cuda/cudnn versions at https://hub.docker.com/r/pytorch/pytorch/tags
ARG PYTORCH="2.11.0"
# cuda12.8, not 12.6: the cu126 wheels top out at sm_90, so Blackwell cards (sm_120, e.g. RTX 50xx)
# load fine and then fail every kernel launch with cudaErrorNoKernelImageForDevice.
ARG CUDA="12.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS dev
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0 12.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV DEBIAN_FRONTEND=noninteractive
# The image ships Ubuntu's python3.12, which is PEP 668 externally-managed. A container is already
# an isolated environment, so install into it directly rather than nesting a venv (a venv would also
# need --system-site-packages to see the prebuilt torch in /usr/local/lib/python3.12).
# Set as env rather than a flag so the editable reinstall in docker/run inherits it too.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# libgl1 + libglib2.0-0t64: shared libs opencv-python links against (24.04 t64 naming).
# ffmpeg: inference_realesrgan_video.py shells out to the binary.
# git: setup.py reads the git hash when generating realesrgan/version.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /Real-ESRGAN

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Not a Real-ESRGAN dependency, so deliberately not in requirements.txt: it exists only so
# docker/jupyter has a server to launch.
RUN pip install --no-cache-dir jupyterlab

# basicsr and facexlib import torchvision.transforms.functional_tensor, which was removed in
# torchvision 0.17. rgb_to_grayscale now lives in torchvision.transforms.functional; the module
# rename is the entire fix. `|| true` so this is a no-op once upstream ships a corrected release.
RUN SP="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && grep -rl 'torchvision\.transforms\.functional_tensor' "$SP" \
       | xargs -r sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g' \
    || true

COPY . ./
RUN pip install --no-cache-dir -e .

# Fail the build here rather than shipping an image that dies on the first import.
RUN python -c "from realesrgan import RealESRGANer; from realesrgan.archs.srvgg_arch import SRVGGNetCompact"
