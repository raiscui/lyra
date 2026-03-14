"""sdg 推理脚本输出命名回归测试."""

from __future__ import annotations

import hashlib

from cosmos_predict1.diffusion.inference.output_naming import (
    build_legacy_sdg_clip_name,
    build_sdg_clip_name,
    has_custom_video_save_name,
    sanitize_clip_name_fragment,
)


def test_sanitize_clip_name_fragment_removes_spaces_and_punctuation() -> None:
    sanitized = sanitize_clip_name_fragment("My demo, 视频!!!  01", max_length=64)

    assert sanitized == "My_demo_视频_01"
    assert " " not in sanitized
    assert "," not in sanitized
    assert "!" not in sanitized


def test_build_sdg_clip_name_uses_image_stem_and_prompt_hash_by_default() -> None:
    prompt = "in the style of Makoto Shinkai,注意镜头移动时候"
    expected_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]

    clip_name = build_sdg_clip_name(
        video_save_name="output",
        visual_input_path="assets/demo/static/diffusion_input/images/xhc.png",
        prompt=prompt,
        batch_input_path=None,
        index=0,
    )

    assert clip_name == f"xhc_{expected_hash}"
    assert " " not in clip_name
    assert "," not in clip_name


def test_build_sdg_clip_name_prefers_explicit_video_save_name() -> None:
    clip_name = build_sdg_clip_name(
        video_save_name="Makoto Demo! 最终版",
        visual_input_path="assets/demo/static/diffusion_input/images/xhc.png",
        prompt="this prompt should not leak into filename",
        batch_input_path=None,
        index=0,
    )

    assert clip_name == "Makoto_Demo_最终版"


def test_build_sdg_clip_name_adds_batch_index_without_prompt_hash() -> None:
    clip_name = build_sdg_clip_name(
        video_save_name="output",
        visual_input_path="assets/demo/static/diffusion_input/images/xhc.png",
        prompt="prompt A",
        batch_input_path="batch.jsonl",
        index=3,
    )

    assert clip_name == "xhc_3"


def test_build_legacy_sdg_clip_name_keeps_old_prompt_suffix() -> None:
    clip_name = build_legacy_sdg_clip_name(
        visual_input_path="assets/demo/static/diffusion_input/images/xhc.png",
        prompt="prompt A",
        batch_input_path=None,
        index=0,
    )

    assert clip_name == "xhc_prompt A"


def test_has_custom_video_save_name_ignores_default_placeholder() -> None:
    assert has_custom_video_save_name(None) is False
    assert has_custom_video_save_name("") is False
    assert has_custom_video_save_name("output") is False
    assert has_custom_video_save_name("my_demo") is True
