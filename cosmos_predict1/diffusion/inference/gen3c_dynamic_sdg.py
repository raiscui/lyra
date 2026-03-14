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
import copy
import torch
import random
import numpy as np
from typing import Dict, Any
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)
from cosmos_predict1.diffusion.inference.output_naming import (
    build_legacy_sdg_clip_name,
    build_sdg_clip_name,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache4D
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from cosmos_predict1.diffusion.inference.data_loader_utils import load_data_auto_detect
from cosmos_predict1.diffusion.inference.vipe_utils import load_vipe_data
import torch.nn.functional as F
torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    ) # TODO: do we need this?
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
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--vipe_path",
        type=str,
        default=None,
        help="Optional: path to VIPE clip root or the mp4 file under rgb/. If set, load VIPE-formatted data directly.",
    )
    parser.add_argument(
        "--vipe_starting_frame_idx",
        type=int,
        default=0,
        help="Starting frame index within the VIPE rgb mp4 to use as the reference frame.",
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
        "--center_depth_quantile",
        action="store_true",
        help="If set, does not use center depth of 1.0 but quantile, which is needed for raw vipe results.",
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
        "--flip_supervision",
        action="store_true",
        help="If set, this generates flipped camera trajectory supervision videos for all multi camera trajectories (only required for training).",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="如果输出文件已存在,仍然强制重新生成并覆盖. 默认会按进度跳过已完成的环节.",
    )
    return parser.parse_args()

def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

def _build_clip_name(args: argparse.Namespace, current_video_path: str, prompt: str | None, index: int) -> str:
    """构造输出文件名的 key,避免把整段 prompt 直接拼进路径."""

    return build_sdg_clip_name(
        video_save_name=args.video_save_name,
        visual_input_path=current_video_path,
        prompt=prompt,
        batch_input_path=args.batch_input_path,
        index=index,
    )


def _build_legacy_clip_name(args: argparse.Namespace, current_video_path: str, prompt: str | None, index: int) -> str:
    """复现历史命名,用于断点续跑兼容旧产物."""

    return build_legacy_sdg_clip_name(
        visual_input_path=current_video_path,
        prompt=prompt,
        batch_input_path=args.batch_input_path,
        index=index,
    )


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
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
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


def _resolve_output_plan(
    args: argparse.Namespace,
    current_video_path: str,
    prompt: str | None,
    index: int,
) -> tuple[str, dict[str, str], dict[str, bool]]:
    """优先采用新短名,但保留对旧长文件名产物的 resume 能力."""

    clip_name = _build_clip_name(args, current_video_path, prompt, index)
    out_paths = _build_output_paths(args, clip_name)
    status = _get_progress_status(args, out_paths)

    legacy_clip_name = _build_legacy_clip_name(args, current_video_path, prompt, index)
    if args.overwrite_existing or legacy_clip_name == clip_name:
        return clip_name, out_paths, status

    legacy_paths = _build_output_paths(args, legacy_clip_name)
    legacy_status = _get_progress_status(args, legacy_paths)
    if any(status.values()):
        return clip_name, out_paths, status
    if any(legacy_status.values()):
        log.info(
            f"[RESUME] Found legacy-named outputs for clip `{clip_name}`. "
            f"Continuing with existing files under `{legacy_clip_name}`."
        )
        return legacy_clip_name, legacy_paths, legacy_status

    return clip_name, out_paths, status


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
        visual_input_path = args.vipe_path if args.vipe_path is not None else args.input_image_path
        prompts = [{"prompt": args.prompt, "visual_input": visual_input_path}]

    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)

    # -----------------------------
    # 断点续跑: 预扫描进度,尽量避免无意义地重复跑 diffusion.
    # -----------------------------
    work_items: list[dict[str, Any]] = []
    any_need_latent = False
    any_need_video_decode = False

    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        current_video_path = input_dict.get("visual_input", None)

        # 保持原行为: prompt/visual 缺失则跳过.
        if current_prompt is None and args.disable_prompt_upsampler:
            work_items.append({"index": i, "skip_reason": "Prompt is missing."})
            continue
        if current_video_path is None:
            work_items.append({"index": i, "skip_reason": "Visual input is missing."})
            continue

        clip_name, out_paths, status = _resolve_output_plan(args, current_video_path, current_prompt, i)

        if args.overwrite_existing:
            need_pose = True
            need_intrinsics = True
            need_latent = True
            need_rgb = True
        else:
            need_pose = not status["pose"]
            need_intrinsics = not status["intrinsics"]
            need_latent = not status["latent"]
            need_rgb = not status["rgb"]

            # save_buffer 会改变最终 mp4 内容(拼接 warp buffer).
            # 为了保证输出一致性,当 mp4 缺失时强制走完整生成流程.
            if getattr(args, "save_buffer", False) and need_rgb:
                need_latent = True

        can_decode_from_latent = (not need_latent) and need_rgb and status["latent"] and (not getattr(args, "save_buffer", False))

        any_need_latent = any_need_latent or need_latent
        any_need_video_decode = any_need_video_decode or can_decode_from_latent

        work_items.append(
            {
                "index": i,
                "prompt": current_prompt,
                "video_path": current_video_path,
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

    # 仅在确实需要生成 latent(跑 diffusion)时才初始化分布式.
    did_init_distributed = False
    process_group = None
    if any_need_latent and args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()
        did_init_distributed = True

    # 仅在需要生成/解码视频时才初始化 pipeline.
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

    sample_n_frames = pipeline.model.chunk_size if (pipeline is not None and any_need_latent) else None

    for item in work_items:
        if "skip_reason" in item:
            log.critical(f"{item['skip_reason']} skipping world generation.")
            continue

        i = int(item["index"])
        current_prompt = item.get("prompt", None)
        current_video_path = item.get("video_path", None)
        clip_name = item["clip_name"]
        out_paths = item["out_paths"]
        status = item["status"]

        # 覆盖模式下必重跑,否则只有当四类产物都齐全才跳过.
        if not args.overwrite_existing and all(status.values()):
            log.info(f"[RESUME] Outputs already exist, skipping: {out_paths['rgb']}")
            continue

        need_pose = bool(item["need_pose"])
        need_intrinsics = bool(item["need_intrinsics"])
        need_latent = bool(item["need_latent"])
        need_rgb = bool(item["need_rgb"])
        can_decode_from_latent = bool(item["can_decode_from_latent"])

        # 如果只缺 rgb 且 latent 已有,直接解码补齐(不加载输入数据,也不跑 diffusion).
        if (not need_latent) and (not need_pose) and (not need_intrinsics) and need_rgb:
            if not can_decode_from_latent:
                log.critical(
                    "[RESUME] rgb is missing but cannot decode from latent in current settings; "
                    "please rerun with --overwrite_existing."
                )
                continue

            if pipeline is None:
                log.critical("[RESUME] Internal error: pipeline is not initialized for latent decoding.")
                continue

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

            latents_tensor = latents_tensor.to(device=device, dtype=torch.bfloat16)
            video = pipeline._run_tokenizer_decoding(latents_tensor)
            if args.flip_supervision:
                video = np.flip(video, axis=0)

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
            continue

        try:
            if args.vipe_path is not None:
                (
                    image_bchw_float,
                    depth_b1hw,
                    mask_b1hw,
                    initial_w2c_b44,
                    intrinsics_b33,
                ) = load_vipe_data(
                    vipe_root_or_mp4=args.vipe_path,
                    starting_frame_idx=args.vipe_starting_frame_idx,
                    resize_hw=(720, 1280),
                    crop_hw=(704, 1280),
                    num_frames=args.num_video_frames,
                )
            else:
                (
                    image_bchw_float,
                    depth_b1hw,
                    mask_b1hw,
                    initial_w2c_b44,
                    intrinsics_b33,
                ) = load_data_auto_detect(current_video_path)
        except Exception as e:
            log.critical(f"Failed to load visual input from {current_video_path}: {e}")
            continue

        image_bchw_float = image_bchw_float.to(device)
        depth_b1hw = depth_b1hw.to(device)
        mask_b1hw = mask_b1hw.to(device)
        initial_w2c_b44 = initial_w2c_b44.to(device)
        intrinsics_b33 = intrinsics_b33.to(device)

        # Reverse frame order before generation
        if args.flip_supervision:
            image_bchw_float = image_bchw_float.flip(dims=[0])
            depth_b1hw = depth_b1hw.flip(dims=[0])
            mask_b1hw = mask_b1hw.flip(dims=[0])
            initial_w2c_b44 = initial_w2c_b44.flip(dims=[0])
            intrinsics_b33 = intrinsics_b33.flip(dims=[0])

        # -----------------------------
        # 断点续跑: 不需要跑 diffusion(latent 已有)时,只补齐缺失产物.
        # -----------------------------
        if not need_latent:
            try:
                center_depth = torch.quantile(depth_b1hw[0], 0.5) if args.center_depth_quantile else 1.0
                generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                    trajectory_type=args.trajectory,
                    initial_w2c=initial_w2c_b44,
                    initial_intrinsics=intrinsics_b33,
                    num_frames=args.num_video_frames,
                    movement_distance=args.movement_distance,
                    camera_rotation=args.camera_rotation,
                    center_depth=center_depth,
                    device=device.type,
                    **args.camera_gen_kwargs,
                )
            except (ValueError, NotImplementedError) as e:
                log.critical(f"Failed to generate trajectory: {e}")
                continue

            generated_c2ws = generated_w2cs.inverse()
            if args.flip_supervision:
                generated_w2cs = generated_w2cs.flip(dims=[1])

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
                if args.flip_supervision:
                    generated_intrinsics = generated_intrinsics.flip(dims=[1])
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

                latents_tensor = latents_tensor.to(device=device, dtype=torch.bfloat16)
                video = pipeline._run_tokenizer_decoding(latents_tensor)
                if args.flip_supervision:
                    video = np.flip(video, axis=0)

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

            continue

        cache = Cache4D(
            input_image=image_bchw_float.clone(), # [B, C, H, W]
            input_depth=depth_b1hw,       # [B, 1, H, W]
            input_mask=mask_b1hw,         # [B, 1, H, W]
            input_w2c=initial_w2c_b44,  # [B, 4, 4]
            input_intrinsics=intrinsics_b33,# [B, 3, 3]
            filter_points_threshold=args.filter_points_threshold,
            input_format=["F", "C", "H", "W"],
            foreground_masking=args.foreground_masking,
        )

        initial_cam_w2c_for_traj = initial_w2c_b44
        initial_cam_intrinsics_for_traj = intrinsics_b33

        # Generate camera trajectory using the new utility function
        try:
            # Set the center depth to 1.0 for already scaled depth/poses, otherwise use depth to determine it
            center_depth = torch.quantile(depth_b1hw[0], 0.5) if args.center_depth_quantile else 1.0
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=center_depth,
                device=device.type,
                **args.camera_gen_kwargs,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            continue

        if pipeline is None or sample_n_frames is None:
            log.critical("[RESUME] Internal error: pipeline is not initialized for diffusion.")
            continue

        log.info(f"Generating 0 - {sample_n_frames} frames")

        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, 0:sample_n_frames],
            generated_intrinsics[:, 0:sample_n_frames],
            start_frame_idx=0,
        )

        all_rendered_warps = []
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())
        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=image_bchw_float[0].unsqueeze(0).unsqueeze(2),
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

            current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                current_segment_w2cs,
                current_segment_intrinsics,
                start_frame_idx=start_frame_idx,
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
        if args.flip_supervision:
            generated_w2cs = generated_w2cs.flip(dims=[1])
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
            if args.flip_supervision:
                generated_intrinsics = generated_intrinsics.flip(dims=[1])
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
            if args.flip_supervision:
                final_video_to_save = np.flip(final_video_to_save, axis=0)
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
    flip_supervision = args.flip_supervision
    
    # Define trajectories
    args.camera_gen_kwargs = {'radius_x_factor': 0.15, 'radius_y_factor': 0.10, 'num_circles': 2}
    trajectories_list = []
    trajectories = {
        "left": {"traj_idx": 0, "movement_distance_range": [0.2, 0.3]},
        "right": {"traj_idx": 1, "movement_distance_range": [0.2, 0.3]},
        "up": {"traj_idx": 2, "movement_distance_range": [0.1, 0.2]},
        "zoom_out": {"traj_idx": 3, "movement_distance_range": [0.3, 0.4]},
        "zoom_in": {"traj_idx": 4, "movement_distance_range": [0.3, 0.4]},
        "clockwise": {"traj_idx": 5, "movement_distance_range": [0.4, 0.6]},
    }
    trajectories_list.append(trajectories)

    # Add flipped supervision for training
    if flip_supervision:
        num_trajectories = len(trajectories)
        trajectories_flipped = {}
        for traj_k, traj_dict in trajectories.items():
            # Main trajectories (first half of indices)
            traj_dict['flip_supervision'] = False
            # Flipped trajectories (second half of indices)
            traj_dict_flipped = copy.deepcopy(traj_dict)
            traj_dict_flipped["traj_idx"] += num_trajectories
            traj_dict_flipped['flip_supervision'] = True
            trajectories_flipped[traj_k] = traj_dict_flipped
        trajectories_list.append(trajectories_flipped)
    
    # Generate for each trajectory independently
    for trajectories in trajectories_list:
        for traj, traj_dict in trajectories.items():
            args.video_save_folder = os.path.join(video_save_folder, str(traj_dict["traj_idx"]))
            args.trajectory = traj
            args.movement_distance = random.uniform(
                traj_dict["movement_distance_range"][0],
                traj_dict["movement_distance_range"][1]
                ) * args.total_movement_distance_factor
            if flip_supervision:
                args.flip_supervision = traj_dict["flip_supervision"]
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
