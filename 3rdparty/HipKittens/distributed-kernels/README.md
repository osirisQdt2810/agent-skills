# Distributed Producer-Consumer GEMM Kernel


## Start docker

```bash
podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    -v $(pwd):/HipKittens \
    -w /HipKittens \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash
```
## Install dependacies

```terminal
apt-get install -y libopenmpi-dev openmpi-bin
pip install mpi4py
```

## Build and Run

```terminal
cd distributed-kernels/

# Configures the build to fetch Iris, select the ROCm/HIP toolchain and GPU architecture, and compile HIP + ThunderKittens code into Python extension modules via pybind11.
cmake -B build -DDK_BUILD=bf16_gemm
# Implements the actual GPU kernel(s) and pybind11 bindings for the tk_kernel Python module, targeting a specific CDNA architecture.
cmake --build build -j 16

# Runs the kernel (single GPU)
python3 example.py
# Multi-GPU (8 GPUs)
mpirun --allow-run-as-root -np 8 python example.py
```
