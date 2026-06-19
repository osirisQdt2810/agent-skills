
## Llama 1B training

### Setting up environment 

```python
pip install -e .
pip install -e .[train]
```

### Installing kernels

```bash
cd HipKittens/
source env.src
cd HipKittens/training/llama/csrc/
bash setup_kernels.sh
```

### Launching training

To train a new model, construct a config.yaml file at ```train/configs/experiment/```. You can launch using the following script:
```bash
cd HipKittens/training/llama/
CUDA_VISIBLE_DEVICES=0 python train/run.py experiment=example/llama-1b trainer.devices=1        # pytorch
CUDA_VISIBLE_DEVICES=1 python train/run.py experiment=example/llama-1b-aiter trainer.devices=1  # aiter (export USE_ROCM_AITER_ROPE_BACKEND=0)
CUDA_VISIBLE_DEVICES=3 python train/run.py experiment=example/llama-1b-hk trainer.devices=1     # hip
```





