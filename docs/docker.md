# Docker

`docker/Dockerfile` has one target per model family. Each image holds the code at `/root/decision-models` with its `.venv`: CUDA 12.8, torch 2.9.0, Python 3.11.

| target | trains | extras |
|---|---|---|
| `encoder` | UltraBERT, Ettin (`make train-encoder`) | `dev` |
| `decoder` | Qwen3 with LLM-as-Jev (`make train-decoder`) | `dev` |
| `qwen35` | Qwen3.5 decoders | `dev`, `qwen35` (builds causal-conv1d) |

You need an NVIDIA driver r570 or newer and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Build

```bash
make docker-build TARGET=encoder        # or decoder, qwen35
# = docker build -f docker/Dockerfile --target encoder -t decision-models:encoder .
make docker-build TARGET=qwen35 DOCKER_ARGS="--build-arg TORCH_CUDA_ARCH_LIST=9.0"   # one GPU generation: faster compile
make docker-build TARGET=encoder DOCKER_ARGS="--platform linux/arm64"                # Grace Hopper / Grace Blackwell
```

## Run

```bash
make docker-run TARGET=encoder
# = docker run --rm -it --gpus all --ipc=host --shm-size=32g \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v $HF_HOME:/root/.cache/huggingface -v $PWD/runs:/root/decision-models/runs \
#     decision-models:encoder
```

Inside the container, train the same way as on the host:

`make train-encoder CONFIG=`[`configs/train/ultrabert_380m_sigmoid_norm.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_sigmoid_norm.yaml)` GPUS=0`<br>
`make train-decoder CONFIG=`[`configs/train/qwen3_0.6b_lora.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/qwen3_0.6b_lora.yaml)` GPUS=0,1,2,3 ARGS="data.source=Athekunal/llm-jev-reason-long"`

| flag | why |
|---|---|
| `--ipc=host --shm-size=32g` | shared memory for NCCL (multi-GPU) and DataLoader workers |
| `-v $HF_CACHE:/root/.cache/huggingface` | keeps weights, datasets and the FA2 kernel between containers (`HF_CACHE` defaults to `$HF_HOME`, else `~/.cache/huggingface`) |
| `-v runs:...` | checkpoints and logs outlive the container |
| `DOCKER_ARGS="-e HF_TOKEN"` | gated models, pushing to the Hub |

## GPU generations

The configs assume Ampere or newer: bf16 autocast, plus FlashAttention-2 varlen for the encoders. On older GPUs, override both settings:

| GPU | compute capability | settings |
|---|---|---|
| Blackwell (B200, RTX 50) | 10.0, 12.0 | configs as they are |
| Hopper (H100, H200, GH200) | 9.0 | configs as they are |
| Ada (L4, L40S, RTX 4090) | 8.9 | configs as they are |
| Ampere (A100, A10, A6000, RTX 3090) | 8.0, 8.6 | configs as they are (the tested setup) |
| Turing (T4, RTX 2080) | 7.5 | `model.attention=sdpa model.pack_train=false run.mixed_precision=fp16` (encoders), `run.mixed_precision=fp16` (decoders) |

The fp16 settings on Turing have not been tested. On a GPU older than Ampere, `attention=varlen` stops at startup with an error that says to use `sdpa`. Only the A100 setup has been trained end to end. The other rows use the same torch and kernel builds.
