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

import argparse
import os
from pathlib import Path
import cv2
import torch
import random
import numpy as np
from typing import Dict, Any
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    add_moge_arguments,
    check_input_frames,
    load_moge_model,
    validate_args,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
import torch.nn.functional as F
torch.enable_grad(False)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    ) # TODO: do we need this?
    add_moge_arguments(parser)
    parser.add_argument(
        "--input_image_path",
        type=str,
        help="Input image path for generating a single video",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left",
            "right",
            "up",
            "down",
            "zoom_in",
            "zoom_out",
            "clockwise",
            "counterclockwise",
            "none",
        ],
        default="left",
        help="Select a trajectory type from the available options (default: original)",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default="center_facing",
        help="Controls camera rotation during movement: center_facing (rotate to look at center), no_rotation (keep orientation), or trajectory_aligned (rotate in the direction of movement)",
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of the camera from the center of the scene",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    parser.add_argument(
        "--multi_trajectory",
        action="store_true",
        help="If set, do multi-trajectory generation used by the 3DGS decoder.",
    )
    parser.add_argument(
        "--camera_gen_kwargs",
        type=Dict[str, Any],
        default={},
    )
    parser.add_argument(
        "--total_movement_distance_factor",
        type=float,
        default=1.0,
        help="Multiply multi trajectory setup with movement distance factor (larger means more movement but potentially more artifacts)",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="如果输出文件已存在,仍然强制重新生成并覆盖. 默认会按进度跳过已完成的环节.",
    )
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()


def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

def _build_clip_name(args: argparse.Namespace, current_image_path: str, prompt: str | None, index: int) -> str:
    """构造输出文件名的 key.

    这里刻意保持与历史脚本一致,避免改动后导致"找不到旧产物"或输出路径变化.
    - 优先使用 args.input_image_path(老逻辑).
    - 当 batch 模式未提供 args.input_image_path 时,回退到 current_image_path.
    """

    base_path = args.input_image_path or current_image_path
    clip_name = Path(base_path).stem

    # 注意: 原脚本会把 prompt 直接拼进文件名,这里也保持一致.
    if prompt is not None and prompt != "":
        clip_name = f"{clip_name}_{prompt}"

    # batch 模式下为了避免重名,追加 index.
    if args.batch_input_path is not None:
        clip_name = f"{clip_name}_{index}"

    return clip_name


def _build_output_paths(args: argparse.Namespace, clip_name: str) -> dict[str, str]:
    """把本次生成涉及的产物路径集中管理,方便做进度检查/断点续跑."""

    return {
        "pose": os.path.join(args.video_save_folder, "pose", f"{clip_name}.npz"),
        "intrinsics": os.path.join(args.video_save_folder, "intrinsics", f"{clip_name}.npz"),
        "latent": os.path.join(args.video_save_folder, "latent", f"{clip_name}.pkl"),
        "rgb": os.path.join(args.video_save_folder, "rgb", f"{clip_name}.mp4"),
    }


def _is_valid_npz(path: str, expected_num_frames: int | None) -> bool:
    """对 npz 做最基本的可读性与 shape 校验,避免"文件存在但内容损坏"导致误跳过."""

    if not os.path.exists(path):
        return False

    try:
        with np.load(path) as data:
            if "data" not in data or "inds" not in data:
                return False
            if expected_num_frames is not None:
                if int(data["inds"].shape[0]) != int(expected_num_frames):
                    return False
                if int(data["data"].shape[0]) != int(expected_num_frames):
                    return False
    except Exception:
        return False

    return True


def _torch_load_compat(path: str):
    """兼容不同 torch 版本的 torch.load 参数差异."""

    try:
        # 新版 torch 支持 weights_only,并且对非权重对象需显式关闭.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 旧版 torch 没有 weights_only 参数.
        return torch.load(path, map_location="cpu")


def _is_valid_latent_pkl(path: str) -> bool:
    """对 latent 做轻量校验: 能否加载 + 维度是否符合[B,C,T,H,W]."""

    if not os.path.exists(path):
        return False

    try:
        latents = _torch_load_compat(path)
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)
        if not isinstance(latents, torch.Tensor):
            return False
        if latents.ndim != 5:
            return False
        # 训练侧会做 latents[0] 去 batch,因此这里要求 batch=1.
        if latents.shape[0] != 1:
            return False
        if latents.numel() == 0:
            return False
    except Exception:
        return False

    return True


def _is_valid_mp4(path: str) -> bool:
    """mp4 只做存在性+文件大小校验(避免 0 字节文件)."""

    if not os.path.exists(path):
        return False
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _get_progress_status(args: argparse.Namespace, out_paths: dict[str, str]) -> dict[str, bool]:
    """汇总每类产物是否已完成."""

    expected = int(args.num_video_frames) if getattr(args, "num_video_frames", None) is not None else None
    return {
        "pose": _is_valid_npz(out_paths["pose"], expected_num_frames=expected),
        "intrinsics": _is_valid_npz(out_paths["intrinsics"], expected_num_frames=expected),
        "latent": _is_valid_latent_pkl(out_paths["latent"]),
        "rgb": _is_valid_mp4(out_paths["rgb"]),
    }


def _predict_moge_depth(current_image_path: str | np.ndarray,
                        target_h: int, target_w: int,
                        device: torch.device, moge_model: Any):
    """Handles MoGe depth prediction for a single image.

    If the image is directly provided as a NumPy array, it should have shape [H, W, C],
    where the channels are RGB and the pixel values are in [0..255].
    """

    if isinstance(current_image_path, str):
        input_image_bgr = cv2.imread(current_image_path)
        if input_image_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path}")
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_image_rgb = current_image_path
    del current_image_path

    depth_pred_h, depth_pred_w = 720, 1280

    input_image_for_depth_resized = cv2.resize(input_image_rgb, (depth_pred_w, depth_pred_h))
    input_image_for_depth_tensor_chw = torch.tensor(input_image_for_depth_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    moge_output_full = moge_model.infer(input_image_for_depth_tensor_chw)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_intrinsics_33_full_normalized = moge_output_full["intrinsics"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(moge_mask_hw_full==0, torch.tensor(1000.0, device=moge_depth_hw_full.device), moge_depth_hw_full)
    moge_intrinsics_33_full_pixel = moge_intrinsics_33_full_normalized.clone()
    moge_intrinsics_33_full_pixel[0, 0] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 1] *= depth_pred_h
    moge_intrinsics_33_full_pixel[0, 2] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 2] *= depth_pred_h

    # Calculate scaling factor for height
    height_scale_factor = target_h / depth_pred_h
    width_scale_factor = target_w / depth_pred_w

    # Resize depth map, mask, and image tensor
    # Resizing depth: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_depth_hw = F.interpolate(
        moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    # Resizing mask: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_mask_hw = F.interpolate(
        moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(target_h, target_w),
        mode='nearest',  # Using nearest neighbor for binary mask
    ).squeeze(0).squeeze(0).to(torch.bool)

    # Resizing image tensor: (C, H, W) -> (1, C, H, W) for interpolate, then squeeze
    input_image_tensor_chw_target_res = F.interpolate(
        input_image_for_depth_tensor_chw.unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    moge_image_b1chw_float = input_image_tensor_chw_target_res.unsqueeze(0).unsqueeze(1) * 2 - 1

    moge_intrinsics_33 = moge_intrinsics_33_full_pixel.clone()
    # Adjust intrinsics for resized height
    moge_intrinsics_33[1, 1] *= height_scale_factor  # fy
    moge_intrinsics_33[1, 2] *= height_scale_factor  # cy
    moge_intrinsics_33[0, 0] *= width_scale_factor  # fx
    moge_intrinsics_33[0, 2] *= width_scale_factor  # cx

    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=1e4)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=1e4)
    moge_mask_b11hw = moge_mask_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # Prepare initial intrinsics [B, 1, 3, 3]
    moge_intrinsics_b133 = moge_intrinsics_33.unsqueeze(0).unsqueeze(0)
    initial_w2c_44 = torch.eye(4, dtype=torch.float32, device=device)
    moge_initial_w2c_b144 = initial_w2c_44.unsqueeze(0).unsqueeze(0)

    return (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        moge_mask_b11hw,
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    )

def _predict_moge_depth_from_tensor(
    image_tensor_chw_0_1: torch.Tensor, # Shape (C, H_input, W_input), range [0,1]
    moge_model: Any
):
    """Handles MoGe depth prediction from an image tensor."""
    moge_output_full = moge_model.infer(image_tensor_chw_0_1)
    moge_depth_hw_full = moge_output_full["depth"]      # (moge_inf_h, moge_inf_w)
    moge_mask_hw_full = moge_output_full["mask"]        # (moge_inf_h, moge_inf_w)

    moge_depth_11hw = moge_depth_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.nan_to_num(moge_depth_11hw, nan=1e4)
    moge_depth_11hw = torch.clamp(moge_depth_11hw, min=0, max=1e4)
    moge_mask_11hw = moge_mask_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.where(moge_mask_11hw==0, torch.tensor(1000.0, device=moge_depth_11hw.device), moge_depth_11hw)

    return moge_depth_11hw, moge_mask_11hw

def demo(args):
    """Run video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先解析输入列表,用于做"断点续跑"的进度扫描.
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": args.prompt, "visual_input": args.input_image_path}]

    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)

    # -----------------------------
    # 断点续跑: 预扫描进度,尽量避免无意义地重复跑 diffusion.
    # -----------------------------
    work_items: list[dict[str, Any]] = []
    any_need_latent = False
    any_need_video_decode = False
    any_need_pose_or_intrinsics = False

    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        current_image_path = input_dict.get("visual_input", None)

        # 保持原行为: prompt/visual 缺失则跳过.
        if current_prompt is None and args.disable_prompt_upsampler:
            work_items.append({"index": i, "skip_reason": "Prompt is missing."})
            continue
        if current_image_path is None:
            work_items.append({"index": i, "skip_reason": "Visual input is missing."})
            continue

        clip_name = _build_clip_name(args, current_image_path, current_prompt, i)
        out_paths = _build_output_paths(args, clip_name)
        status = _get_progress_status(args, out_paths)

        if args.overwrite_existing:
            # 覆盖模式: 强制从头生成,并覆盖所有产物.
            need_pose = True
            need_intrinsics = True
            need_latent = True
            need_rgb = True
        else:
            # 默认模式: 以产物是否已存在来决定是否跳过.
            need_pose = not status["pose"]
            need_intrinsics = not status["intrinsics"]
            need_latent = not status["latent"]
            need_rgb = not status["rgb"]

            # save_buffer 会改变最终 mp4 内容(拼接 warp buffer).
            # 为了保证输出一致性,当 mp4 缺失时强制走完整生成流程.
            if getattr(args, "save_buffer", False) and need_rgb:
                need_latent = True

        # 仅当 "latent 已有但 rgb 缺失" 且不需要 save_buffer 时,才允许走解码补齐.
        can_decode_from_latent = (not need_latent) and need_rgb and status["latent"] and (not getattr(args, "save_buffer", False))

        any_need_latent = any_need_latent or need_latent
        any_need_video_decode = any_need_video_decode or can_decode_from_latent
        any_need_pose_or_intrinsics = any_need_pose_or_intrinsics or need_pose or need_intrinsics

        work_items.append(
            {
                "index": i,
                "prompt": current_prompt,
                "image_path": current_image_path,
                "clip_name": clip_name,
                "out_paths": out_paths,
                "status": status,
                "need_pose": need_pose,
                "need_intrinsics": need_intrinsics,
                "need_latent": need_latent,
                "need_rgb": need_rgb,
                "can_decode_from_latent": can_decode_from_latent,
            }
        )

    # 如果没有任何需要补齐的产物,直接退出(避免重复加载大模型).
    any_need_work = any(
        (
            (item.get("need_pose") or item.get("need_intrinsics") or item.get("need_latent") or item.get("need_rgb"))
            for item in work_items
            if "skip_reason" not in item
        )
    )
    if not args.overwrite_existing and not any_need_work:
        log.info("[RESUME] All outputs already exist. Nothing to do, exiting.")
        return

    # -----------------------------
    # 仅在确实需要生成 latent(跑 diffusion)时才初始化分布式.
    # -----------------------------
    did_init_distributed = False
    process_group = None
    if any_need_latent and args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()
        did_init_distributed = True

    # -----------------------------
    # 仅在需要生成/解码视频时才初始化 pipeline.
    # - 如果只是 decode(不跑 diffusion),强制 offload_network,避免加载巨大 DiT.
    # -----------------------------
    pipeline: Gen3cPipeline | None = None
    if any_need_latent or any_need_video_decode:
        pipeline_offload_network = args.offload_diffusion_transformer if any_need_latent else True
        pipeline = Gen3cPipeline(
            inference_type=inference_type,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name="Gen3C-Cosmos-7B",
            prompt_upsampler_dir=args.prompt_upsampler_dir,
            enable_prompt_upsampler=not args.disable_prompt_upsampler,
            offload_network=pipeline_offload_network,
            offload_tokenizer=args.offload_tokenizer,
            offload_text_encoder_model=args.offload_text_encoder_model,
            offload_prompt_upsampler=args.offload_prompt_upsampler,
            offload_guardrail_models=args.offload_guardrail_models,
            disable_guardrail=args.disable_guardrail,
            disable_prompt_encoder=args.disable_prompt_encoder,
            guidance=args.guidance,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_video_frames=121,
            seed=args.seed,
        )

        if any_need_latent and args.num_gpus > 1:
            pipeline.model.net.enable_context_parallel(process_group)

    # 只有需要跑 diffusion 时才需要 MoGe + cache 相关常量.
    frame_buffer_max = pipeline.model.frame_buffer_max if (pipeline is not None and any_need_latent) else None
    sample_n_frames = pipeline.model.chunk_size if (pipeline is not None and any_need_latent) else None
    generator = torch.Generator(device=device).manual_seed(args.seed) if any_need_latent else None
    # MoGe v1/v2 都实现了 `.infer(...)`,这里用 Any 作为最小接口,避免脚本固定绑死某个版本.
    moge_model: Any | None = None

    for item in work_items:
        # 预扫描阶段已判断为必跳过的输入,这里保持原行为: 打 log 后 continue.
        if "skip_reason" in item:
            log.critical(f"{item['skip_reason']} skipping world generation.")
            continue

        i = int(item["index"])
        current_prompt = item.get("prompt", None)
        current_image_path = item.get("image_path", None)
        clip_name = item["clip_name"]
        out_paths = item["out_paths"]
        status = item["status"]

        # 覆盖模式下必重跑,否则只有当四类产物都齐全才跳过.
        if not args.overwrite_existing and all(status.values()):
            log.info(f"[RESUME] Outputs already exist, skipping: {out_paths['rgb']}")
            continue

        # 按需决定本轮要补齐哪些产物.
        need_pose = bool(item["need_pose"])
        need_intrinsics = bool(item["need_intrinsics"])
        need_latent = bool(item["need_latent"])
        need_rgb = bool(item["need_rgb"])
        can_decode_from_latent = bool(item["can_decode_from_latent"])

        # Check input frames
        if not check_input_frames(current_image_path, 1):
            print(f"Input image {current_image_path} is not valid, skipping.")
            continue

        # -----------------------------
        # 断点续跑: 本轮不需要跑 diffusion(latent 已有)时,只补齐缺失产物.
        # -----------------------------
        if not need_latent:
            # 1) pose / intrinsics 缺失: 仅重建相机轨迹并补齐,不跑 diffusion.
            if need_pose or need_intrinsics:
                if moge_model is None:
                    # MoGe 只在需要"从图像估计深度/内参"时才加载,避免无意义的显存占用.
                    moge_model, _moge_version = load_moge_model(
                        moge_version=args.moge_version,
                        moge_model_id=args.moge_model_id,
                        moge_checkpoint_path=args.moge_checkpoint_path,
                        hf_local_files_only=args.hf_local_files_only,
                        device=device,
                    )

                (
                    _moge_image_b1chw_float,
                    _moge_depth_b11hw,
                    _moge_mask_b11hw,
                    moge_initial_w2c_b144,
                    moge_intrinsics_b133,
                ) = _predict_moge_depth(
                    current_image_path, args.height, args.width, device, moge_model
                )

                initial_cam_w2c_for_traj = moge_initial_w2c_b144[0, 0]
                initial_cam_intrinsics_for_traj = moge_intrinsics_b133[0, 0]

                try:
                    generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                        trajectory_type=args.trajectory,
                        initial_w2c=initial_cam_w2c_for_traj,
                        initial_intrinsics=initial_cam_intrinsics_for_traj,
                        num_frames=args.num_video_frames,
                        movement_distance=args.movement_distance,
                        camera_rotation=args.camera_rotation,
                        center_depth=1.0,
                        device=device.type,
                        **args.camera_gen_kwargs,
                    )
                except (ValueError, NotImplementedError) as e:
                    log.critical(f"Failed to generate trajectory: {e}")
                    continue

                generated_c2ws = generated_w2cs.inverse()

                if need_pose:
                    pose_save_path = out_paths["pose"]
                    os.makedirs(os.path.dirname(pose_save_path), exist_ok=True)
                    pose_list = []
                    for f_idx in range(generated_c2ws.shape[1]):
                        pose = generated_c2ws[0, f_idx].cpu().numpy().reshape(4, 4)
                        pose_list.append((f_idx, pose))
                    pose_data = np.stack([pose for _, pose in pose_list], axis=0)
                    pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
                    np.savez(pose_save_path, data=pose_data, inds=pose_inds)
                    log.info(f"[RESUME] Saved pose to {pose_save_path}")

                if need_intrinsics:
                    intrinsics_save_path = out_paths["intrinsics"]
                    os.makedirs(os.path.dirname(intrinsics_save_path), exist_ok=True)
                    intrinsics_list = []
                    for f_idx in range(generated_intrinsics.shape[1]):
                        intrinsics = generated_intrinsics[0, f_idx].cpu().numpy()
                        intrinsics_fxfycxcy = (
                            intrinsics[0, 0],
                            intrinsics[1, 1],
                            intrinsics[0, 2],
                            intrinsics[1, 2],
                        )
                        intrinsics_list.append((f_idx, intrinsics_fxfycxcy))
                    intrinsics_data = np.stack([intrinsics for _, intrinsics in intrinsics_list], axis=0)
                    intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])
                    np.savez(intrinsics_save_path, data=intrinsics_data, inds=intrinsics_inds)
                    log.info(f"[RESUME] Saved intrinsics to {intrinsics_save_path}")

            # 2) rgb 缺失,且 latent 已有: 直接解码补齐,避免重新跑 diffusion.
            if need_rgb:
                if not can_decode_from_latent:
                    log.critical(
                        "[RESUME] rgb is missing but cannot decode from latent in current settings; "
                        "please rerun with --overwrite_existing."
                    )
                    continue

                if pipeline is None:
                    log.critical("[RESUME] Internal error: pipeline is not initialized for latent decoding.")
                    continue

                # 如果 tokenizer 被 offload 了,这里按需加载回来.
                if getattr(pipeline.model, "tokenizer", None) is None:
                    pipeline._load_tokenizer()

                latents_obj = _torch_load_compat(out_paths["latent"])
                if isinstance(latents_obj, np.ndarray):
                    latents_tensor = torch.from_numpy(latents_obj)
                else:
                    latents_tensor = latents_obj

                if not isinstance(latents_tensor, torch.Tensor):
                    log.critical(f"[RESUME] Failed to load latent tensor: {out_paths['latent']}")
                    continue

                # tokenizer decode 需要 latent 在 GPU 上(与 pipeline.model 的 device 对齐).
                latents_tensor = latents_tensor.to(device=device, dtype=torch.bfloat16)
                video = pipeline._run_tokenizer_decoding(latents_tensor)

                rgb_save_path = out_paths["rgb"]
                os.makedirs(os.path.dirname(rgb_save_path), exist_ok=True)
                save_video(
                    video=video,
                    fps=args.fps,
                    H=int(video.shape[1]),
                    W=int(video.shape[2]),
                    video_save_quality=8,
                    video_save_path=rgb_save_path,
                )
                log.info(f"[RESUME] Decoded video from latent and saved to {rgb_save_path}")

            # 本轮不需要跑 diffusion,处理完缺失产物后直接进入下一个输入.
            continue

        # -----------------------------
        # 需要生成 latent: 走完整 diffusion 生成流程.
        # -----------------------------
        if pipeline is None or frame_buffer_max is None or generator is None or sample_n_frames is None:
            log.critical("[RESUME] Internal error: pipeline/cache constants are not initialized for diffusion.")
            continue

        if moge_model is None:
            # MoGe 仅在需要跑 diffusion 或补齐相机轨迹时才加载.
            moge_model, _moge_version = load_moge_model(
                moge_version=args.moge_version,
                moge_model_id=args.moge_model_id,
                moge_checkpoint_path=args.moge_checkpoint_path,
                hf_local_files_only=args.hf_local_files_only,
                device=device,
            )

        # load image, predict depth and initialize 3D cache
        (
            moge_image_b1chw_float,
            moge_depth_b11hw,
            moge_mask_b11hw,
            moge_initial_w2c_b144,
            moge_intrinsics_b133,
        ) = _predict_moge_depth(
            current_image_path, args.height, args.width, device, moge_model
        )

        cache = Cache3D_Buffer(
            frame_buffer_max=frame_buffer_max,
            generator=generator,
            noise_aug_strength=args.noise_aug_strength,
            input_image=moge_image_b1chw_float[:, 0].clone(), # [B, C, H, W]
            input_depth=moge_depth_b11hw[:, 0],       # [B, 1, H, W]
            # input_mask=moge_mask_b11hw[:, 0],         # [B, 1, H, W]
            input_w2c=moge_initial_w2c_b144[:, 0],  # [B, 4, 4]
            input_intrinsics=moge_intrinsics_b133[:, 0],# [B, 3, 3]
            filter_points_threshold=args.filter_points_threshold,
            foreground_masking=args.foreground_masking,
        )

        initial_cam_w2c_for_traj = moge_initial_w2c_b144[0, 0]
        initial_cam_intrinsics_for_traj = moge_intrinsics_b133[0, 0]

        # Generate camera trajectory using the new utility function
        try:
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=1.0,
                device=device.type,
                **args.camera_gen_kwargs,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            continue

        log.info(f"Generating 0 - {sample_n_frames} frames")
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, 0:sample_n_frames],
            generated_intrinsics[:, 0:sample_n_frames],
        )

        all_rendered_warps = []
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=current_image_path,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
            return_latents=True,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt, latents = generated_output

        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(1, num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1) # Overlap by 1 frame
            end_frame_idx = start_frame_idx + sample_n_frames

            log.info(f"Generating {start_frame_idx} - {end_frame_idx} frames")

            last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
            pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0 # (C,H,W), range [0,1]

            pred_depth, pred_mask = _predict_moge_depth_from_tensor(
                pred_image_for_depth_chw_0_1, moge_model
            )

            cache.update_cache(
                new_image=pred_image_for_depth_chw_0_1.unsqueeze(0) * 2 - 1, # (B,C,H,W) range [-1,1]
                new_depth=pred_depth, #  (1,1,H,W)
                # new_mask=pred_mask,   # (1,1,H,W)
                new_w2c=generated_w2cs[:, start_frame_idx],
                new_intrinsics=generated_intrinsics[:, start_frame_idx],
            )
            current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                current_segment_w2cs,
                current_segment_intrinsics,
            )

            if args.save_buffer:
                all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())


            pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1 # (B,C,T,H,W), range [-1,1]
            generated_output = pipeline.generate(
                prompt=current_prompt,
                image_path=pred_image_for_depth_bcthw_minus1_1,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
                return_latents=True,
            )
            video_new, prompt, latents_new = generated_output
            video = np.concatenate([video, video_new[1:]], axis=0)
            latents = torch.cat([latents, latents_new[1:]], axis=0)

        # Final video processing
        final_video_to_save = video
        final_width = args.width

        if args.save_buffer and all_rendered_warps:
            squeezed_warps = [t.squeeze(0) for t in all_rendered_warps] # Each is (T_chunk, n_i, C, H, W)

            if squeezed_warps:
                n_max = max(t.shape[1] for t in squeezed_warps)

                padded_t_list = []
                for sq_t in squeezed_warps:
                    # sq_t shape: (T_chunk, n_i, C, H, W)
                    current_n_i = sq_t.shape[1]
                    padding_needed_dim1 = n_max - current_n_i

                    pad_spec = (0,0, # W
                                0,0, # H
                                0,0, # C
                                0,padding_needed_dim1, # n_i
                                0,0) # T_chunk
                    padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                    padded_t_list.append(padded_t)

                full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)

                T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
                buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
                buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
                buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
                buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
                buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))

                final_video_to_save = np.concatenate([buffer_numpy_THWC, final_video_to_save], axis=2)
                final_width = args.width * (1 + n_max)
                log.info(f"Concatenating video with {n_max} warp buffers. Final video width will be {final_width}")
            else:
                log.info("No warp buffers to save.")

        # Save pose
        generated_c2ws = generated_w2cs.inverse()
        if need_pose:
            pose_save_path = out_paths["pose"]
            os.makedirs(os.path.dirname(pose_save_path), exist_ok=True)
            pose_list = []
            for f_idx in range(generated_c2ws.shape[1]):
                pose = generated_c2ws[0, f_idx].cpu().numpy().reshape(4, 4)
                pose_list.append((f_idx, pose))
            pose_data = np.stack([pose for _, pose in pose_list], axis=0)
            pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
            np.savez(pose_save_path, data=pose_data, inds=pose_inds)
            log.info(f"[RESUME] Saved pose to {pose_save_path}")
        else:
            log.info(f"[RESUME] Pose already exists, skipping: {out_paths['pose']}")

        # Save intrinsics
        if need_intrinsics:
            intrinsics_save_path = out_paths["intrinsics"]
            os.makedirs(os.path.dirname(intrinsics_save_path), exist_ok=True)
            intrinsics_list = []
            for f_idx in range(generated_intrinsics.shape[1]):
                intrinsics = generated_intrinsics[0, f_idx].cpu().numpy()
                intrinsics_fxfycxcy = (
                    intrinsics[0, 0],
                    intrinsics[1, 1],
                    intrinsics[0, 2],
                    intrinsics[1, 2],
                )
                intrinsics_list.append((f_idx, intrinsics_fxfycxcy))
            intrinsics_data = np.stack([intrinsics for _, intrinsics in intrinsics_list], axis=0)
            intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])
            np.savez(intrinsics_save_path, data=intrinsics_data, inds=intrinsics_inds)
            log.info(f"[RESUME] Saved intrinsics to {intrinsics_save_path}")
        else:
            log.info(f"[RESUME] Intrinsics already exists, skipping: {out_paths['intrinsics']}")

        # Save latent
        if need_latent:
            latent_save_path = out_paths["latent"]
            os.makedirs(os.path.dirname(latent_save_path), exist_ok=True)
            video_latent = latents.detach().float().cpu().numpy()
            torch.save(video_latent, latent_save_path)
            log.info(f"[RESUME] Saved latent to {latent_save_path}")
        else:
            log.info(f"[RESUME] Latent already exists, skipping: {out_paths['latent']}")

        # Save rgb video
        if need_rgb:
            video_save_path = out_paths["rgb"]
            os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
            save_video(
                video=final_video_to_save,
                fps=args.fps,
                H=int(final_video_to_save.shape[1]),
                W=int(final_video_to_save.shape[2]),
                video_save_quality=8,
                video_save_path=video_save_path,
            )
            log.info(f"[RESUME] Saved video to {video_save_path}")
        else:
            log.info(f"[RESUME] Video already exists, skipping: {out_paths['rgb']}")

    # clean up properly
    if did_init_distributed:
        # 只有当本次确实 init 了分布式,才执行 destroy,避免误报错.
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()

def demo_multi_trajectory(args):
    video_save_folder = args.video_save_folder
    # Define trajectories
    args.camera_gen_kwargs = {'radius_x_factor': 0.15, 'radius_y_factor': 0.10, 'num_circles': 2}
    trajectories = {
        "left": {"traj_idx": 0, "movement_distance_range": [0.2, 0.3]},
        "right": {"traj_idx": 1, "movement_distance_range": [0.2, 0.3]},
        "up": {"traj_idx": 2, "movement_distance_range": [0.1, 0.2]},
        "zoom_out": {"traj_idx": 3, "movement_distance_range": [0.3, 0.4]},
        "zoom_in": {"traj_idx": 4, "movement_distance_range": [0.3, 0.4]},
        "clockwise": {"traj_idx": 5, "movement_distance_range": [0.4, 0.6]},
    }
    # Generate for each trajectory independently
    for traj, traj_dict in trajectories.items():
        args.video_save_folder = os.path.join(video_save_folder, str(traj_dict["traj_idx"]))
        args.trajectory = traj
        args.movement_distance = random.uniform(
            traj_dict["movement_distance_range"][0],
            traj_dict["movement_distance_range"][1]
            ) * args.total_movement_distance_factor
        demo(args)

if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    if args.multi_trajectory:
        demo_multi_trajectory(args)
    else:
        demo(args)
