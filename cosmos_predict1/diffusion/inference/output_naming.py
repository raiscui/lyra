"""sdg 推理输出命名工具.

这部分逻辑只依赖标准库。
这样一来,纯字符串命名测试就不需要把整个推理栈都 import 进来。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

DEFAULT_SINGLE_VIDEO_SAVE_NAME = "output"
SAFE_CLIP_NAME_PATTERN = re.compile(r"[^\w-]+", flags=re.UNICODE)
SAFE_CLIP_NAME_REPEAT_UNDERSCORE_PATTERN = re.compile(r"_+")


def has_custom_video_save_name(video_save_name: str | None) -> bool:
    """判断用户是否真的显式指定了输出名."""

    return video_save_name not in (None, "", DEFAULT_SINGLE_VIDEO_SAVE_NAME)


def sanitize_clip_name_fragment(value: str | None, *, max_length: int = 48) -> str:
    """把任意文本清洗成适合落盘的短文件名片段.

    这里保留 Unicode 单词字符(含中文),但去掉空格和大多数标点,
    避免生成 `foo, bar ???.mp4` 这类可读性和兼容性都很差的路径。
    """

    if value is None:
        return DEFAULT_SINGLE_VIDEO_SAVE_NAME

    sanitized = SAFE_CLIP_NAME_PATTERN.sub("_", value).strip("._-")
    sanitized = SAFE_CLIP_NAME_REPEAT_UNDERSCORE_PATTERN.sub("_", sanitized)
    if not sanitized:
        return DEFAULT_SINGLE_VIDEO_SAVE_NAME

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("._-")

    return sanitized or DEFAULT_SINGLE_VIDEO_SAVE_NAME


def build_sdg_clip_name(
    *,
    video_save_name: str | None,
    visual_input_path: str,
    prompt: str | None,
    batch_input_path: str | None,
    index: int,
) -> str:
    """给 sdg 系列入口生成短且稳定的 clip name.

    规则分两层:
    - 用户显式给了 `--video_save_name` 时,优先尊重它,只做安全清洗。
    - 否则回退到输入视觉资产 stem,并在单条 prompt 模式下追加短 hash,
      既避免长文件名,也保留“同图不同 prompt 不重名”的能力。
    """

    if has_custom_video_save_name(video_save_name):
        clip_name = sanitize_clip_name_fragment(video_save_name, max_length=64)
    else:
        clip_name = sanitize_clip_name_fragment(Path(visual_input_path).stem, max_length=48)
        normalized_prompt = (prompt or "").strip()
        if batch_input_path is None and normalized_prompt:
            prompt_hash = hashlib.sha1(normalized_prompt.encode("utf-8")).hexdigest()[:8]
            clip_name = f"{clip_name}_{prompt_hash}"

    if batch_input_path is not None:
        clip_name = f"{clip_name}_{index}"

    return clip_name


def build_legacy_sdg_clip_name(
    *,
    visual_input_path: str,
    prompt: str | None,
    batch_input_path: str | None,
    index: int,
) -> str:
    """复现 sdg 脚本旧版命名,用于 resume 兼容历史产物."""

    clip_name = Path(visual_input_path).stem
    normalized_prompt = (prompt or "").strip()
    if normalized_prompt:
        clip_name = f"{clip_name}_{normalized_prompt}"

    if batch_input_path is not None:
        clip_name = f"{clip_name}_{index}"

    return clip_name
