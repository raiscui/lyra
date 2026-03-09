# LATER_PLANS


## 2026-03-04 13:32 UTC

- 为 `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py` 增加"外部视频输入"模式:
  - 支持 `--input_video_path` 直接读取外部 mp4.
  - 用 tokenizer encoder(`model.encode`)把外部视频编码成 `latent/*.pkl`.
  - pose/intrinsics 优先支持从外部 `*.npz` 载入;若不提供,才使用脚本当前的 `generate_camera_trajectory(...)`(并提示用户必须保证相机运动匹配).
- 核对并可能修复长视频模式的 latent 拼接维度:
  - 当前脚本对 `latents` 使用 `torch.cat(..., axis=0)` 的拼接方式可疑,需要通过一次实际运行打印 shape 来确认时序维度约定.

## 2026-03-04 14:25 UTC

- 跟进 Warp 1.7.2 在 CUDA 13 driver 下的 `cuDeviceGetUuid` 初始化告警:
  - 目标: 确认是否仅为噪音,或是否会影响长时间运行/多进程场景的稳定性.
  - 若影响: 评估升级 warp 或从源码按当前 CUDA driver/toolkit 重新构建.


## 2026-03-06 04:49 UTC

- 如果后续目标从"当前单轨迹去重影"扩展为"无可靠 pose 的外部长视频 / 多段轨迹增量融合",可以单开一条 `LongSplat-style` 路线:
  - 引入新帧注册(PnP + pose refinement)
  - 增加 local/global sliding window schedule
  - 研究 visibility ratio 驱动的窗口推进
  - 评估 anchor/octree 表示是否值得替换当前高斯组织方式


## 2026-03-06 05:37 UTC

- 如果后续决定把“超分对等视频”真正落地到现有 Lyra 流程,优先做成 `sample.py` 之后的独立 post-refinement 阶段,不要先走“把超分视频直接回灌 feed-forward 主链”:
  - 输入建议:
    - `gaussians_init_ply`
    - 原 `pose/*.npz`
    - 原 `intrinsics/*.npz`
    - 超分后的对等视频
  - 默认策略:
    - 先 `gaussian-only refinement`
    - 先外观/透明度,后有限几何
    - 默认不先碰 pose
  - 关键前提:
    - 超分视频必须保持同一 crop / aspect / 时序
    - intrinsics 要按超分倍率同步缩放
    - 优先做 patch-based 高分辨率监督,避免整帧显存爆炸

## 2026-03-06 07:28 UTC

- 下一轮优先继续补强 `Stage 2A` 后的质量项:
  - opacity/pruning
  - patch-based supervision
  - 如有必要再进入 `Stage 2B`

## 2026-03-06 08:33 UTC

- `refinement_v2` 的 `opacity/pruning` 已完成落地并做过真实验证,后续待办里不再把它当“未实现项”.
- 下一轮优先级更新为:
  1. patch-based supervision
  2. 如 patch 监督后仍存在明显重影,再评估更强的 `stage2b` 或更激进的 pruning 调度

## 2026-03-06 09:12 UTC

- `patch-based supervision` 的第一版已经完成,后续待办不再是“是否实现 patch path”,而是:
  1. 是否接入真正的外部 SR/reference 视频到 `reference_images`
  2. 是否继续推进 `stage2b` 来补锐度与几何层问题
  3. 是否把当前推荐参数更新为 `lambda_patch_rgb=0.25` 作为困难轨迹默认起点

## 2026-03-06 07:28 UTC

- 在 `Long-LRM style post refinement` 主线稳定后,可以评估两个明确的后续分支:
  1. `Mip-Splatting` 的 renderer-level `2D Mip filter`
     - 当前只吸收了 `3D smoothing` 思想
     - 如果后续 selective SR 稳定后仍存在明显 aliasing / dilation,再研究是否需要改造 rasterization 行为
  2. `EDGS-style local reinitialization`
     - 当前只保留为 deferred idea
     - 只有在 Stage 3A + Stage 3SR 后仍有局部结构缺失,并且证据更像初始化覆盖不足时,才考虑做局部 dense correspondence + triangulation re-seed

## 2026-03-06 10:37 UTC

- `Stage 2B` 本身已经完成,后续待办不再是“是否实现 Stage 2B”,而是更偏调参与增强方向:
  1. 评估 `Stage 2B` 从原始高斯直接启动时,是否需要独立的更长 `iters_stage2a` 或不同 gate 统计项.
  2. 把“从已验证 Stage 2A 基线续跑 Stage 2B”整理成一个显式 CLI/workflow 模式.
  3. 继续推进真正的外部 SR/reference 视频接入,让 `Stage 2B` 能吃到真实高分参考而不是仅 native patch.

## 2026-03-06 11:02 UTC

- 如果后续要继续榨 `Stage 2B` warm-start 的极限质量,可以评估是否增加一个显式 warmup 选项:
  - 例如 `stage2a_warmup_iters_before_stage2b`
- 动机:
  - 旧手工路线比新 `--start-stage stage2b` 略强
  - 证据更像是“进入 Stage 2B 前多跑了 1 轮 Stage 2A”带来的额外收益

## 2026-03-06 11:36 UTC

- external reference contract 已经完成,因此后续待办应从“能不能接外部 reference”转成更深一层的两个方向:
  1. full direct file inputs
     - `--pose-path`
     - `--intrinsics-path`
     - `--rgb-path`
     - 不依赖 dataloader 直接构造 `SceneBundle`
  2. selective SR 主线补全
     - `gaussian_fidelity_score`
     - `W_sr_select`
     - `W_final_sr`
     - `L_sampling_smooth`
     - `Phase 3S / Stage 3SR`

## 2026-03-08 07:04 UTC

- 如果后续要真正解锁 full-view `target_subsample=4`, 优先研究 `src/rendering/gs.py` 的 `render_meta` 内存路径:
  1. 只保留 refinement 当前真正需要的 meta key
  2. 改成按阶段按需 materialize, 不在 `Phase 0` 全量 `cat + stack`
  3. 评估能否把部分统计改成 chunk 内归约, 而不是保留完整 dense tensor
- 动机:
  - A800 80GB 上两轮 `sub4` smoke 都在 `_merge_chunk_meta()` OOM
  - 说明单纯换 allocator 已不足以支撑这档 observation 密度

## 2026-03-09 04:20 UTC

- 如果后续继续推进 `add-refinement-v2-depth-anchor`,优先采用更保守的落地顺序:
  1. baseline reference 复用 `Phase 0 / Phase 1` evaluation render,不要先改 dataloader depth 契约
  2. V1 先观察 `Stage 3SR` 单独启用 depth anchor 的收益与副作用
  3. 只有在证据表明不会明显锁死 baseline 厚表面时,再扩到 `Stage 2A`
