# Docker

A CUDA-enabled development image for Real-ESRGAN.

```bash
./docker/build          # builds real-esrgan:local, then verifies GPU kernels
./docker/run            # interactive shell, repo bind-mounted at /Real-ESRGAN
./docker/jupyter        # JupyterLab on http://localhost:8888, serving the repo root
```

`build` runs from any directory; `run` and `jupyter` must be started from the repository root, since they
bind-mount `$(pwd)`. See `./docker/build --help` for the full option list. `run` reads `DATASETS_DIR` (default `/mnt/c/DATASETS`) and
mounts it at `/Real-ESRGAN/DATASETS`; if the directory does not exist the mount is skipped with a warning.
`jupyter` prints an access token on startup — copy it from the terminal to log in. `jupyterlab` is
installed in the image rather than in `requirements.txt`, since it is not a Real-ESRGAN dependency.

Because the repo is bind-mounted, edits on the host take effect inside the container immediately — no
rebuild needed for code changes. Rebuild only when `requirements.txt`, the base image, or the Dockerfile
itself changes.

## Updating the image

Version pins live in three `ARG`s at the top of `Dockerfile`. Available tags are listed at
<https://hub.docker.com/r/pytorch/pytorch/tags>.

```dockerfile
ARG PYTORCH="2.11.0"
ARG CUDA="12.8"
ARG CUDNN="9"
```

To try a combination without editing the Dockerfile, override on the command line — and give it a
separate tag so the working image is not clobbered:

```bash
./docker/build --cuda 13.0 --tag real-esrgan:cu130
./docker/build --pytorch 2.13.0 --cuda 12.6      # cu126: expect the GPU check to fail on Blackwell
```

Any arg you do not pass keeps the Dockerfile's default. Edit the `ARG`s once a combination is proven, so
the default is the one that works.

Confirm the combination actually exists before rebuilding — a wrong triple fails only after the build
starts:

```bash
docker manifest inspect pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel > /dev/null && echo "tag exists"
```

To list candidate tags:

```bash
curl -s "https://hub.docker.com/v2/repositories/pytorch/pytorch/tags?page_size=100&name=devel" \
  | python3 -c "import sys,json; [print(t['name']) for t in json.load(sys.stdin)['results']]"
```

**Do not go below `cuda12.8`.** The `cu126` wheels build kernels only up to `sm_90`. On a Blackwell card
(`sm_120`, e.g. RTX 50xx) the image still reports `torch.cuda.is_available() == True` and then fails every
kernel launch with `cudaErrorNoKernelImageForDevice`. Newer CUDA is fine; older is a silent trap.

If you add an architecture, extend `TORCH_CUDA_ARCH_LIST` to match. It only affects CUDA extensions
compiled inside the container — the prebuilt torch ignores it — but a stale list is worse than none.

## Testing a rebuild

Run these after any change. Each catches a distinct failure mode. Step 1 is automatic and catches most of
them; steps 2 and 3 cover what only shows up once the repo is bind-mounted over the image.

**1. Build, which covers imports and GPU kernels.** Do not pipe to `tail`/`head` — the pipeline returns
the pager's exit code, so a failed build reports success:

```bash
./docker/build > /tmp/build.log 2>&1; echo "BUILD_EXIT=$?"
```

`BUILD_EXIT=0` is the pass condition, and it covers two of the three failure modes on its own. The
Dockerfile's last stage is an import guard, so a green build proves the Python environment is intact.
`build` then runs a GPU kernel check, because `torch.cuda.is_available()` is not sufficient — it returns
`True` on an unsupported architecture. A passing check looks like:

```
Verifying GPU kernels...
  device:      NVIDIA GeForce RTX 5060
  capability:  sm_120
  arch list:   sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
  matmul:      ok
real-esrgan:local ready
```

Your card's capability must appear in the arch list, and the matmul must run rather than raise. The check
is skipped automatically when no GPU runtime is available, and can be skipped explicitly with
`--no-verify`.

**2. Imports resolve against the bind-mount.** This is the path `docker/run` uses, and it differs from the
build-time environment — mounting the repo hides files the build generated:

```bash
docker run --rm --ipc=host -v "$(pwd):/Real-ESRGAN" real-esrgan:local /bin/bash -c '
  pip install -q --no-deps -e . 2>/dev/null
  python -c "import realesrgan, cv2, basicsr, gfpgan; from realesrgan import RealESRGANer; print(\"ok\")"'
```

**3. End-to-end inference.** Downloads ~64 MB of weights on first run into `weights/`:

```bash
docker run --rm --gpus all --ipc=host -v "$(pwd):/Real-ESRGAN" real-esrgan:local \
  python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/0014.jpg -o results --outscale 4
```

`results/0014_out.jpg` should be exactly 4× the input (179×179 → 716×716). Note that files written by the
container are owned by root; `chown -R --reference=setup.py results weights` from inside the container
fixes that, and `docker/run` already does it for the one file that matters.

## Why the Dockerfile looks the way it does

Each of these is load-bearing. Removing one produces an image that builds and then fails at runtime.

| Line | Reason |
|---|---|
| `ENV PIP_BREAK_SYSTEM_PACKAGES=1` | The image's `python` is Ubuntu 24.04's system `python3.12`, marked PEP 668 externally-managed; PyTorch is just COPY'd into `/usr/local/lib/python3.12`. Plain `pip install` fails with `externally-managed-environment`. A venv would need `--system-site-packages` to see the prebuilt torch, which is more moving parts than it's worth in a container. |
| The `functional_tensor` `sed` | `basicsr` and `facexlib` import `torchvision.transforms.functional_tensor`, removed in torchvision 0.17. `rgb_to_grayscale` moved to `torchvision.transforms.functional`; the module rename is the whole fix. Without it `import basicsr` raises and every entry point is dead. The `\|\| true` makes it a no-op once upstream ships a corrected release. |
| `libgl1`, `libglib2.0-0t64` | Shared libraries `opencv-python` links against. Missing them, `import cv2` fails with `libGL.so.1: cannot open shared object file`. On Ubuntu 24.04 the glib package carries the `t64` suffix — plain `libglib2.0-0` does not exist there. |
| `ffmpeg` | `inference_realesrgan_video.py` shells out to the binary. |
| `git` | `setup.py` reads the git hash while generating `realesrgan/version.py`. |
| Final `RUN python -c "from realesrgan import ..."` | Turns a broken environment into a failed build instead of a runtime surprise. |

There is deliberately **no conda**. These PyTorch images have been conda-free since roughly 2.4 — Ubuntu
24.04, Python 3.12, no `/opt/conda`, nothing named `conda` on `PATH`. Older versions of this Dockerfile
had `conda update conda` and an `environment-dev.yml`; both were removed because they cannot work on a
modern base. There is likewise no nvidia GPG-key workaround: that block targeted a 2022-era image, and its
`rm /etc/apt/sources.list.d/cuda.list` fails outright here because the file does not exist.
