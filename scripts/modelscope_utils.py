# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 注意:
# - 这里选择通过 ModelScope CLI 进行下载, 而不是直接调用 Python API.
# - 原因是 CLI 原生支持 `--local_dir` + `--include/--exclude`, 可以更接近 Hugging Face `allow_patterns` 的行为,
#   并且更容易保持本仓库既有的 `checkpoints/` 目录结构不变.

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence


def modelscope_download(
    model_id: str,
    local_dir: str | Path,
    *,
    revision: Optional[str] = None,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    files: Optional[Sequence[str]] = None,
) -> None:
    """
    下载 ModelScope Hub 上的模型到指定目录.

    参数说明:
    - model_id: ModelScope 的模型 ID, 形如 "nv-community/GEN3C-Cosmos-7B".
    - local_dir: 下载到的本地目录. 我们用它来稳定落到 `checkpoints/...`.
    - revision: 可选, 指定分支/版本.
    - include/exclude: glob 过滤, 用于复刻 Hugging Face 的 allow_patterns.
    - files: 指定仓库内相对路径的文件列表(和 include/exclude 二选一).
    """

    modelscope_cli = shutil.which("modelscope")
    if modelscope_cli is None:
        raise RuntimeError(
            "未找到 `modelscope` 命令. 请先安装依赖: `python -m pip install modelscope`."
        )

    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        modelscope_cli,
        "download",
        "--model",
        model_id,
        "--local_dir",
        str(local_dir_path),
    ]
    if revision:
        cmd += ["--revision", revision]

    # files 与 include/exclude 互斥.
    if files:
        cmd += list(files)
    else:
        if include:
            cmd += ["--include", *list(include)]
        if exclude:
            cmd += ["--exclude", *list(exclude)]

    try:
        # 让 CLI 直接把进度打印出来, 便于用户观察下载过程.
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "无法执行 `modelscope` 命令. 请确认 modelscope 已正确安装, 且命令在 PATH 中."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ModelScope 下载失败: {model_id}. 你可以删除目录后重试: {local_dir_path}"
        ) from e


def ensure_list(value: Optional[Iterable[str]]) -> list[str]:
    """把可选的可迭代对象规整成 list, 方便上层组装 include/exclude 参数."""

    if value is None:
        return []
    return list(value)

