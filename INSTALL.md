### Environment setup

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.10.x`.

We manage the environment with [pixi](https://pixi.sh). The workspace is defined in `pixi.toml`.
```bash
# Create / update the pixi environment (conda dependencies).
pixi install

# Enter the environment.
pixi shell

# (Optional) If you have connectivity issues, configure a proxy first.
# Example (Clash default ports):
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897

# Install the dependencies.
python -m pip install -r requirements_gen3c.txt
python -m pip install -r requirements_lyra.txt

# (Optional) Install Transformer Engine (TE).
# - Cosmos can run without TE (it will fall back to the PyTorch attention backend).
# - If you need TE acceleration, install prebuilt wheels to avoid long compilation time.
#   1) Core package (prebuilt on PyPI):
#      python -m pip install "transformer-engine==1.12.0"
#   2) PyTorch extension (install a prebuilt wheel that matches your Python/Torch/CUDA versions):
#      python -m pip install /path/to/transformer_engine_torch-*.whl
 pip install --no-build-isolation transformer_engine[pytorch]
# Install Apex for inference.
test -d apex || git clone https://github.com/NVIDIA/apex
CUDA_HOME="$CONDA_PREFIX" python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex

# Install MoGe for inference.
python -m pip install git+https://github.com/microsoft/MoGe.git

# Install Mamba for reconstruction model.
python -m pip install --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"
```

You can test the environment setup for inference with
```bash
CUDA_HOME="$CONDA_PREFIX" PYTHONPATH="$(pwd)" python scripts/test_environment.py
```

Optionally, check training-specific dependencies as well:
```bash
CUDA_HOME="$CONDA_PREFIX" PYTHONPATH="$(pwd)" python scripts/test_environment.py --training
```

### Download Cosmos-Predict1 tokenizer

Tokenizer checkpoints are hosted on ModelScope:
- https://modelscope.cn/models/nv-community/Cosmos-Tokenize1-CV8x8x8-720p

Download the Cosmos Tokenize model weights from ModelScope:
```bash
python3 -m scripts.download_tokenizer_checkpoints --checkpoint_dir checkpoints/cosmos_predict1 --tokenizer_types CV8x8x8-720p
```

Optionally, download guardrail checkpoints (not required if you run with `--disable_guardrail`):
```bash
python3 -m scripts.download_tokenizer_checkpoints --checkpoint_dir checkpoints/cosmos_predict1 --tokenizer_types CV8x8x8-720p --download_guardrail
```
Note: the guardrail download includes `meta-llama/Llama-Guard-3-8B`, which is currently fetched from Hugging Face Hub.

The downloaded files should be in the following structure:
```
checkpoints/
├── Cosmos-Tokenize1-CV8x8x8-720p
├── Cosmos-Tokenize1-DV8x16x16-720p
├── Cosmos-Tokenize1-CI8x8-360p
├── Cosmos-Tokenize1-CI16x16-360p
├── Cosmos-Tokenize1-CV4x8x8-360p
├── Cosmos-Tokenize1-DI8x8-360p
├── Cosmos-Tokenize1-DI16x16-360p
└── Cosmos-Tokenize1-DV4x8x8-360p
```

Under the checkpoint repository `checkpoints/<model-name>`, we provide the encoder, decoder, the full autoencoder in TorchScript (PyTorch JIT mode) and the native PyTorch checkpoints. For instance for `Cosmos-Tokenize1-CV8x8x8-720p` model:
```bash
├── checkpoints/
│   ├── Cosmos-Tokenize1-CV8x8x8-720p/
│   │   ├── encoder.jit
│   │   ├── decoder.jit
│   │   ├── autoencoder.jit
│   │   ├── model.pt
```

### Download GEN3C checkpoints

GEN3C checkpoints are hosted on ModelScope:
- https://modelscope.cn/models/nv-community/GEN3C-Cosmos-7B

Download the GEN3C model weights:
   ```bash
   CUDA_HOME="$CONDA_PREFIX" PYTHONPATH="$(pwd)" python scripts/download_gen3c_checkpoints.py --checkpoint_dir checkpoints
   ```

### Download Lyra checkpoints

Lyra checkpoints are hosted on ModelScope:
- https://modelscope.cn/models/nv-community/Lyra

Download the Lyra model weights:
   ```bash
   CUDA_HOME="$CONDA_PREFIX" PYTHONPATH="$(pwd)" python scripts/download_lyra_checkpoints.py --checkpoint_dir checkpoints
   ```
