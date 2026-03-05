# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List

from scripts.modelscope_utils import modelscope_download


def download_models(models: List[str], destination_root: str):
    """
    Download guardrail-related models and save them in org/project structure.

    Args:
        models: List of model IDs in format 'org/project'
        destination_root: Root directory where models will be saved
    """
    for model_id in models:
        model_id, revision = model_id.split(":") if ":" in model_id else (model_id, None)
        print(f"Downloading {model_id}...")

        # Create the full path for the model
        model_path = os.path.join(destination_root, model_id)

        try:
            # Cosmos-Guardrail1 已在 ModelScope 上以 nv-community 组织提供.
            if model_id == "nvidia/Cosmos-Guardrail1":
                modelscope_download(
                    "nv-community/Cosmos-Guardrail1",
                    model_path,
                    revision=revision,
                )
            else:
                # 其余 guardrail 组件(例如 Llama-Guard)仍从 Hugging Face Hub 下载.
                # 为了不让下载脚本在 import 阶段就硬依赖 huggingface_hub,这里使用惰性导入.
                try:
                    from huggingface_hub import snapshot_download  # type: ignore
                except ModuleNotFoundError as e:
                    raise RuntimeError(
                        "缺少依赖 `huggingface_hub`,无法下载部分 guardrail 权重(例如 Llama-Guard). "
                        "请先安装: `python -m pip install huggingface_hub`."
                    ) from e

                snapshot_download(
                    repo_id=model_id,
                    local_dir=model_path,
                    revision=revision,
                )
            print(f"Successfully downloaded {model_id} to {model_path}")

        except Exception as e:
            raise RuntimeError(f"Error downloading {model_id}: {str(e)}. Please delete the directory and try again.")


def download_guardrail_checkpoints(destination_root: str):
    """
    Download guardrail checkpoints and save them in org/project structure.

    Args:
        destination_root: Root directory where checkpoints will be saved
    """
    # List of models to download
    models_to_download = [
        "meta-llama/Llama-Guard-3-8B",
        "nvidia/Cosmos-Guardrail1",
    ]

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_root, exist_ok=True)

    # Download the models
    download_models(models_to_download, destination_root)


if __name__ == "__main__":
    download_guardrail_checkpoints("checkpoints")
