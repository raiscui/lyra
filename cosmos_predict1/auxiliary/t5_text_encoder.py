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

from typing import List, Tuple, Union

import os
import torch
import transformers
from transformers import T5EncoderModel, T5TokenizerFast

transformers.logging.set_verbosity_error()

#
# T5 prompt encoder 的默认模型信息集中放在这里.
# 这样无论是直接实例化编码器,还是上层 pipeline 调用,都不会再分叉到不同目录.
#
DEFAULT_T5_MODEL_NAME = "google-t5/t5-11b"
DEFAULT_T5_MODEL_DIR = "/model/HuggingFace/google-t5/t5-11b"


class CosmosT5TextEncoder(torch.nn.Module):
    """Handles T5 text encoding operations."""

    def __init__(
        self,
        model_name: str = DEFAULT_T5_MODEL_NAME,
        device: str = "cuda",
        cache_dir: str = DEFAULT_T5_MODEL_DIR,
    ):
        """Initializes the T5 tokenizer and encoder.

        Args:
            model_name: The name of the T5 model to use.
            device: The device to use for computations.
            cache_dir: 本地模型目录路径. 这里不会自动联网下载.
        """
        super().__init__()
        model_dir = os.path.expanduser(cache_dir)
        try:
            # 只从本地目录加载,避免误触发在线下载.
            # 如果目录不存在,直接给出明确提示,避免 users 在长栈里找不到原因.
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(model_dir)

            self.tokenizer = T5TokenizerFast.from_pretrained(model_dir, local_files_only=True)
            self.text_encoder = T5EncoderModel.from_pretrained(model_dir, local_files_only=True).to(device)
        except Exception as e:
            raise RuntimeError(
                f"无法从本地目录加载 T5 prompt encoder({model_name}): {model_dir}.\n"
                "请先把 Hugging Face 模型 `google-t5/t5-11b` 下载到该目录.\n"
                "参考页面: https://huggingface.co/google-t5/t5-11b.\n"
                "如果你在容器里运行,请确认已把宿主机模型目录挂载到该路径.\n"
                f"底层错误: {e}"
            ) from e
        self.text_encoder.eval()
        self.device = device

    @torch.inference_mode()
    def encode_prompts(
        self, prompts: Union[str, List[str]], max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes text prompts into hidden state representations using a T5 encoder.

        This function tokenizes the input prompts, processes them through a T5 text encoder,
        and returns the last hidden states. The encoded outputs beyond the actual sequence
        length are zero-padded. All prompts in a batch are padded to max_length.

        Args:
            prompts: Input text to encode. Can be a single string or a list of strings.
            max_length: Maximum sequence length for tokenization and padding. Longer
                sequences will be truncated. Defaults to 512.
            return_mask: If True, returns the attention mask along with encoded text.
                Defaults to False.

        Returns:
            If return_mask is False:
                torch.Tensor: Encoded text embeddings of shape (batch_size, max_length, hidden_size).
            If return_mask is True:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - Encoded text embeddings of shape (batch_size, max_length, hidden_size)
                    - Attention mask of shape (batch_size, max_length) as boolean tensor

        Raises:
            ValueError: If the input prompts list is empty.

        Example:
            >>> encoder = CosmosT5TextEncoder(cache_dir=DEFAULT_T5_MODEL_DIR)
            >>> prompts = ["Hello world", "Another example"]
            >>> embeddings = encoder.encode_prompts(prompts, max_length=128)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("The input prompt list is empty.")

        batch_encoding = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )

        input_ids = batch_encoding.input_ids.to(self.device)
        attn_mask = batch_encoding.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

        encoded_text = outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()

        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        return encoded_text, attn_mask


class DummyT5TextEncoder(torch.nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    @torch.inference_mode()
    def encode_prompts(
        self, prompts: Union[str, List[str]], max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("The input prompt list is empty.")

        batch_size = len(prompts)
    
        dummy_text_embedding = torch.zeros(batch_size, max_length, 1024, device=self.device)
        dummy_text_mask = torch.zeros(batch_size, max_length, device=self.device, dtype=torch.bool)
        dummy_text_mask[0] = True

        return dummy_text_embedding, dummy_text_mask
