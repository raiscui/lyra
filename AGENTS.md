# Repository Guidelines

## Project Structure

- `src/`: core Lyra implementation (models, data, rendering, utils).
- `cosmos_predict1/`: Cosmos/GEN3C components used for diffusion + tokenization.
- `configs/`: YAML configs for demo, inference, and training (e.g. `configs/demo/lyra_static.yaml`).
- `scripts/`: checkpoint download + environment checks; `scripts/bash/` contains SDG launch scripts.
- `assets/`: small inputs and examples. Large generated outputs typically live under `assets/demo/` (gitignored).
- `checkpoints/`, `lyra_dataset/`, `apex/`: downloaded/compiled artifacts (do not commit).

## Setup, Demo, and Common Commands

This repo is tested on Linux with Python 3.10. The Pixi workspace is defined in `pixi.toml`.

```bash
pixi install
pixi shell
python -m pip install -r requirements_gen3c.txt
# NOTE: `flash-attn==2.6.3` 在 torch 2.6 环境下通常需要从源码编译,并依赖 `cicc`.
PATH="$CONDA_PREFIX/nvvm/bin:$PATH" CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux" python -m pip install -r requirements_lyra.txt
CUDA_HOME="$CONDA_PREFIX" PYTHONPATH="$(pwd)" python scripts/test_environment.py
```

Run the demo reconstruction (3DGS decoder) via `accelerate`:

```bash
accelerate launch sample.py --config configs/demo/lyra_static.yaml
```

Other entry points:
- `bash train.sh`: progressive training.
- `bash inference.sh`: export renderings/gaussians during training.

## Coding Style & Naming

- Python: 4-space indentation, prefer small functions, and keep imports tidy.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes.
- Prefer improving existing modules over adding new ones. Keep diffs focused and documented.

## Testing Guidelines

- Smoke test: `scripts/test_environment.py` (run after dependency, CUDA, or checkpoint loader changes).
- Pytest tests exist as `*_test.py` under `cosmos_predict1/`. Example:
  `PYTHONPATH="$(pwd)" pytest -v cosmos_predict1/tokenizer/modules/layers2d_test.py`

## Commits & Pull Requests

- Commit subjects are short and imperative (e.g. `Update README.md`).
- All commits must be signed off for DCO compliance: `git commit -s -m "..."`.
- Open PRs as Draft until ready for review. Include hardware/CUDA version, config path(s) used, and logs or sample outputs when relevant.
