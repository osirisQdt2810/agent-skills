
## Baseline kernels

This README describes the baseline kernels we use from third party libraries at the time of this work (October 2025). We benchmarked all kernels using 500 warmup and 100 repeat iterations. 

**Warning:** Many software packages are under continuous development. We provide instructions as of 11/09/2025. 

### Composable kernel

**Attention**

Baselines were collected using this process:
```bash
[~] git clone https://github.com/rocm/composable_kernel
[~] cd composable_kernel
[~/composable_kernel] mkdir build && cd build
[~/composable_kernel/build] ../script/cmake-ck-dev.sh .. gfx950 -G Ninja
[~/composable_kernel/build] ninja tile_example_gemm_basic
```

Just in case, here is a working commit at the time of this work [commit](https://github.com/ROCm/composable_kernel/tree/d88ea05c844cd159a14213b73a5818a43c5b79e6).

For attention, we ran the above for ```ninja tile_example_fmha_fwd``` and ```ninja tile_example_fmha_bwd```. Then we benchmarked with the following commands. We only found a ```ck_tile``` example in the repository at the time of this work (October 2025):

Forwards:
```bash
# non-causal forwards mha
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal forwards mha
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1

# non-causal forwards gqa
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal forwards gqa
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1
```

Backwards:
```bash
# non-causal mha
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal mha
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1

# non-causal gqa
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal gqa
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1 
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1
```

**GEMM**

From within ghte ```composable_kernel/build/``` directory run:
```bash
ninja tile_example_gemm_basic
ninja tile_example_gemm_universal
ninja tile_example_streamk_gemm_basic
```

We picked the best of these options for each dimension:
```bash 
# https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/40_streamk_gemm 
./bin/tile_example_streamk_gemm_basic -prec=bf16 -m=1024 -n=1024 -k=1024 -warmup=500 -repeat=100 -v=1 
./bin/tile_example_streamk_gemm_basic -prec=bf16 -m=2048 -n=2048 -k=2048 -warmup=500 -repeat=100 -v=1 
./bin/tile_example_streamk_gemm_basic -prec=bf16 -m=4096 -n=4096 -k=4096 -warmup=500 -repeat=100 -v=1 
./bin/tile_example_streamk_gemm_basic -prec=bf16 -m=8192 -n=8192 -k=8192 -warmup=500 -repeat=100 -v=1 
./bin/tile_example_streamk_gemm_basic -prec=bf16 -m=16384 -n=16384 -k=16384 -warmup=500 -repeat=100 -v=1 

# https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/03_gemm
# *NOTE* We can run with `bf16` or `fp8` dtypes.
./bin/tile_example_gemm_basic -prec=fp16 -m=1024 -n=1024 -k=1024 -warmup=500 -repeat=100 -v=1   
./bin/tile_example_gemm_basic -prec=fp16 -m=2048 -n=2048 -k=2048 -warmup=500 -repeat=100 -v=1
./bin/tile_example_gemm_basic -prec=fp16 -m=4096 -n=4096 -k=4096 -warmup=500 -repeat=100 -v=1
./bin/tile_example_gemm_basic -prec=fp16 -m=8192 -n=8192 -k=8192 -warmup=500 -repeat=100 -v=1
./bin/tile_example_gemm_basic -prec=fp16 -m=16384 -n=16384 -k=16384 -warmup=500 -repeat=100 -v=1

./bin/tile_example_gemm_universal -prec=fp16 -m=1024 -n=1024 -k=1024 -warmup=500 -repeat=100 -v=1
./bin/tile_example_gemm_universal -prec=fp16 -m=2048 -n=2048 -k=2048 -warmup=500 -repeat=100 -v=1
./bin/tile_example_gemm_universal -prec=fp16 -m=4096 -n=4096 -k=4096 -warmup=500 -repeat=100 -v=1
./bin/tile_example_gemm_universal -prec=fp16 -m=8192 -n=8192 -k=8192 -warmup=500 -repeat=100 -v=1
./bin/tile_example_gemm_universal -prec=fp16 -m=16384 -n=16384 -k=16384 -warmup=500 -repeat=100 -v=1
```

**Low precision GEMM**
The build is ```ninja example_gemm_mx_fp6```:
```bash
./bin/example_gemm_mx_fp6 0 2 1 0 1024 1024 1024 -1 -1 -1 1 500 100
./bin/example_gemm_mx_fp6 0 2 1 0 4096 4096 4096 -1 -1 -1 1 500 100
```


### Triton baselines

First ```pip install matplotlib```.

**Attention**

Attention baselines for triton are taken as the best performance out of:
- [ROCm triton perf-kernels](https://github.com/ROCm/triton/tree/76076e1d7d16a988a61a66264845990acd1244ab/python/perf-kernels) ```flash-attention.py```
- [ROCm triton tutorials](https://github.com/ROCm/triton/tree/76076e1d7d16a988a61a66264845990acd1244ab/python/tutorials) ```06-fused-attention.py```

We can directly run the files using python. Find these files under. **Set causality, fwd/bwd, and gqa/mha appropriately in the file**. We report the best of:
```bash
cd baselines/attn/
python baselines/attn/triton_attention_v01.py # does not support GQA
python baselines/attn/triton_attention_v02.py # only causal bwd is supported. 
 ```
Note some of the shapes lead to OOMs.


**GEMM**

We report the best of the following implementations for each dimension. These kernels are pulled from [ROCm triton perf-kernels](https://github.com/ROCm/triton/tree/76076e1d7d16a988a61a66264845990acd1244ab/python/perf-kernels) and [ROCm triton tutorials](https://github.com/ROCm/triton/tree/76076e1d7d16a988a61a66264845990acd1244ab/python/tutorials). 
```bash
cd baselines/attn/
python baselines/attn/triton_gemm_v01.py
python baselines/attn/triton_gemm_v02.py
python baselines/attn/triton_gemm_v03.py
 ```


### HipblasLT baselines

**GEMM**

BF16 GEMM:
```bash
# Note the hpl and norm_dist perform basically the same. We use torch.randn in PyTorch so consider norm_dist here. 
hipblaslt-bench --batch_count 1 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --rotating 512 --iters 100 --cold_iters 500 -m 1024 -n 1024 -k 1024 --initialization norm_dist
hipblaslt-bench --batch_count 1 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --rotating 512 --iters 100 --cold_iters 500 -m 2048 -n 2048 -k 2048 --initialization norm_dist
hipblaslt-bench --batch_count 1 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --rotating 512 --iters 100 --cold_iters 500 -m 4096 -n 4096 -k 4096 --initialization norm_dist
hipblaslt-bench --batch_count 1 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --rotating 512 --iters 100 --cold_iters 500 -m 8192 -n 8192 -k 8192 --initialization norm_dist
hipblaslt-bench --batch_count 1 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --rotating 512 --iters 100 --cold_iters 500 -m 16384 -n 16384 -k 16384 --initialization norm_dist
```

FP8 GEMM:
```bash
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 100 --cold_iters 500 -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --ldd 8192
```

FP6 GEMM:
```bash
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries/projects/hipBLASLt

sudo apt-get install -y libboost-filesystem-dev libboost-system-dev

# configure
cmake -B build -S .                                  \
      -D CMAKE_BUILD_TYPE=Release                    \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
      -D CMAKE_PREFIX_PATH=/opt/rocm                 \         
      -D GPU_TARGETS=gfx950
# build
cmake --build build --parallel

./install.sh -c -a gfx950

# Run
cd build/release/
./clients/hipblaslt-bench --api_method c -m 1024 -n 1024 -k 1024 --alpha 1 --beta 0 --transA T --transB N --batch_count 1 --scaleA 3 --scaleB 3 --a_type f6_r --b_type f6_r --c_type f16_r --d_type f16_r --compute_type f32_r --rotating 0 --cold_iters 500 --iters 100
./clients/hipblaslt-bench --api_method c -m 2048 -n 2048 -k 2048 --alpha 1 --beta 0 --transA T --transB N --batch_count 1 --scaleA 3 --scaleB 3 --a_type f6_r --b_type f6_r --c_type f16_r --d_type f16_r --compute_type f32_r --rotating 0 --cold_iters 500 --iters 100
./clients/hipblaslt-bench --api_method c -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transA T --transB N --batch_count 1 --scaleA 3 --scaleB 3 --a_type f6_r --b_type f6_r --c_type f16_r --d_type f16_r --compute_type f32_r --rotating 0 --cold_iters 500 --iters 100
./clients/hipblaslt-bench --api_method c -m 8192 -n 8192 -k 8192 --alpha 1 --beta 0 --transA T --transB N --batch_count 1 --scaleA 3 --scaleB 3 --a_type f6_r --b_type f6_r --c_type f16_r --d_type f16_r --compute_type f32_r --rotating 0 --cold_iters 500 --iters 100
./clients/hipblaslt-bench --api_method c -m 16384 -n 16384 -k 16384 --alpha 1 --beta 0 --transA T --transB N --batch_count 1 --scaleA 3 --scaleB 3 --a_type f6_r --b_type f6_r --c_type f16_r --d_type f16_r --compute_type f32_r --rotating 0 --cold_iters 500 --iters 1000
```

## Mojo baselines

Setup the docker and environment (https://docs.modular.com/mojo/manual/get-started/):
```
# This docker recommended by mojo gives errors (as of 11/8/2025).
docker.io/modular/max-amd-base 
# amd nightly is also not supporting rocm 7.0 currently.

# So we instead use beta:
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
    --entrypoint /bin/bash \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta 

# if you don't have it, install pixi
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="/root/.pixi/bin:$PATH"

# create a project
pixi init life \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd life

# install the modular conda package
pixi add modular

# setup the VM
pixi shell
```

Next install the repository:

```bash
git clone https://github.com/modular/modular.git
cd max/kernels/benchmarks/gpu/
```

Run the kernels (you can manually edit these files to choose your desired problem dimensions):
```
mojo bench_mha.mojo
mojo bench_matmul.mojo
```

**Warning:** As of 11/09/2025, we are not sure if the [mha causal numbers are correctly reported](https://github.com/modular/modular/issues/5557).


## TileLang baselines

```bash
cd /workdir/projects/tilelang
rm -rf build
# Set environment for ROCm
export USE_ROCM=/opt/rocm
export USE_CUDA=0
# Install with pip (let it handle the entire build)
pip install -e . -v
# If that fails, try with explicit backend:
pip install -e . --config-settings=cmake.args="-DUSE_CUDA=OFF" -v
```


