# WORKLOG

## 2026-03-04 00:00 UTC: 修复 `scripts/test_environment.py` 的 `megatron.core` 导入失败

- 复现并确认: 在 `.pixi/envs/default` 环境中,`scripts/test_environment.py` 报 `[ERROR] megatron.core`.
- 定位根因: `megatron.core` 导入链路触发 `transformer_engine.pytorch` 加载,但 TE 缺少 `transformer_engine_torch` 扩展,抛出 `StopIteration` 导致导入失败.
- 修复措施: 通过源码编译安装 `transformer_engine_torch==1.12.0`,补齐 TE 的 PyTorch 扩展.
- 验证结果: `CUDA_HOME=/usr/local/cuda PYTHONPATH="$(pwd)" .pixi/envs/default/bin/python scripts/test_environment.py` 全部 `[SUCCESS]`.
- 文档同步: 更新 `INSTALL.md`,明确 "只装 TE core 但不装 `transformer_engine_torch` 会导致 `megatron.core` 导入失败",并移除容易误导的安装命令.
- 仓库清爽: 更新 `.gitignore`,忽略 `apex/`,避免按安装文档 clone 后污染 `git status`.

## 2026-03-04 08:14 UTC: 将 T5 prompt encoder 切换到 ModelScope `mindnlp/t5-11b`

- 运行时加载路径:
  - `cosmos_predict1/utils/base_world_generation_pipeline.py` 默认从 `checkpoints/mindnlp/t5-11b` 加载.
  - 若不存在则回退到 `checkpoints/google-t5/t5-11b`,避免老环境直接崩.
- 下载脚本:
  - `scripts/download_gen3c_checkpoints.py` 改为通过 ModelScope 下载 `mindnlp/t5-11b`,不再走 Hugging Face.
  - 增加了 T5 目录完整性检查(配置/Tokenizer/权重文件是否存在),避免半下载误判成功.
- 避免误触发在线下载:
  - `cosmos_predict1/auxiliary/t5_text_encoder.py` 改为 `local_files_only=True`,只从本地目录加载.
- 文档同步:
  - `INSTALL.md` 补充说明: GEN3C 下载脚本会同时下载 `mindnlp/t5-11b`.
- 验证:
  - `python3 -m compileall` 对相关文件做了语法编译检查,无报错.

## 2026-03-04 08:45 UTC: 修复下载脚本对 `huggingface_hub` 的硬依赖

- 问题表现: 用户运行 `scripts/download_gen3c_checkpoints.py` 时,即使只需要 ModelScope 下载,也会因缺少 `huggingface_hub` 在 import 阶段崩溃.
- 根因: 多个下载脚本顶层 import 了 Hugging Face 相关模块,以及可选 guardrail 模块导致传递依赖.
- 修复:
  - `scripts/download_gen3c_checkpoints.py` / `scripts/download_lyra_checkpoints.py` 改为惰性导入 `huggingface_hub`(仅 Pixtral 转换时需要).
  - `scripts/download_guardrail_checkpoints.py` 改为惰性导入 `huggingface_hub`(仅下载非 ModelScope 的 guardrail 权重时需要).
  - `scripts/download_tokenizer_checkpoints.py` 仅在 `--download_guardrail` 时才 import guardrail 下载逻辑.
- 验证: `PYTHONPATH="$(pwd)" python3 scripts/download_gen3c_checkpoints.py --help` 能正常输出帮助信息.

## 2026-03-04 10:44 UTC: 还原 T5 prompt encoder 为 Hugging Face `google-t5/t5-11b`

- 问题表现: 推理脚本尝试从 `checkpoints/mindnlp/t5-11b` 加载 T5,但该目录缺少 `pytorch_model.bin/model.safetensors` 等 HF 权重文件,导致 `transformers` 报错并中断.
- 根因: `mindnlp/t5-11b` 与 Hugging Face 的 `google-t5/t5-11b` 不是同一个模型/权重形态,不能用同一套 `from_pretrained(<local_dir>)` 逻辑直接加载.
- 修复:
  - 运行时默认只从固定本地目录 `/model/HuggingFace/google-t5/t5-11b` 加载 T5 prompt encoder.
  - 若目录不存在或不完整,直接抛出带指引的错误(不再静默回退到其它来源,避免再次用错模型).
  - `scripts/download_gen3c_checkpoints.py` 不再下载/依赖任何 T5 权重,避免脚本继续误导.
  - `INSTALL.md` 同步更新,明确提示用户从 Hugging Face 准备 `google-t5/t5-11b` 到上述目录.
- 验证:
  - `PYTHONPATH="$(pwd)" .pixi/envs/default/bin/python -m py_compile ...` 通过.
  - `CosmosT5TextEncoder(cache_dir="/this/path/should/not/exist")` 能输出明确的缺失目录提示.

## 2026-03-04 12:58 UTC: 新增 `AGENTS.md` 作为仓库贡献者指南

- 新增 `AGENTS.md`(标题 "Repository Guidelines"),用于快速说明目录结构、环境安装、demo/训练入口与测试方式.
- 命令示例对齐现有文档:
  - `pixi install && pixi shell` + `requirements_*.txt` 安装依赖.
  - `scripts/test_environment.py` 作为 smoke test.
  - `accelerate launch sample.py --config configs/demo/lyra_static.yaml` 作为 demo 示例.
- 贡献约束同步:
  - 强调 `.gitignore` 已忽略 `checkpoints/`、`assets/demo/`、`lyra_dataset/`、`apex/` 等大文件/产物目录,避免误提交.
  - 提醒 DCO sign-off: `git commit -s -m "..."`

## 2026-03-04 13:42 UTC: 降级 `flash-attn` 到 2.6.3 以匹配 Transformer Engine

- 背景: `transformer_engine==1.12.0` 仅支持 `flash-attn<=2.6.3`,环境原有 `flash-attn==2.7.4.post1` 会输出 warning 且 TE 不启用 flash-attn 后端.
- 依赖锁定: `requirements_lyra.txt` 将 `flash_attn` 固定为 `2.6.3`.
- 安装落地: 在 `.pixi/envs/default` 卸载旧版本并从源码编译安装 `flash-attn==2.6.3`(torch 2.6 无对应预编译 wheel).
  - 关键修复: 编译前需要把 `$CONDA_PREFIX/nvvm/bin` 放到 `PATH`,否则会报 `cicc: not found`.
- 验证:
  - `import transformer_engine` 不再打印版本 warning.
  - `transformer_engine.pytorch.attention.flash_attn_func` 成功加载(非 None).
  - `PYTHONPATH="$(pwd)"` 下 `import megatron.core` 正常.

## 2026-03-04 13:32 UTC: 分析 `gen3c_single_image_sdg.py` 的视频生成与产物依赖

- 梳理 `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py` 的主流程: 单图 -> MoGe 深度/内参 -> 生成相机轨迹 -> Gen3C 扩散生成视频(分块自回归) -> 落盘 `rgb/pose/intrinsics/latent`.
- 明确脚本落盘的"其他产物"并非都由 mp4 再推导:
  - `pose/*.npz` 与 `intrinsics/*.npz` 来自 `generate_camera_trajectory(...)` 的轨迹设定(视频生成前就已得到,只是生成后才写盘).
  - `latent/*.pkl` 与视频是同一次扩散采样的伴生产物,本质是 tokenizer latent 空间的最终采样结果(解码后才得到 mp4).
- 追踪下游依赖: `src/models/data/radym.py` 会按固定目录结构读取 `latent/pose/intrinsics` 与 `rgb`,说明仅替换 mp4 往往不够,还需要匹配的 latent 与相机参数.
- 给出替换结论: mp4 可以来自外部工具,但要继续走默认重建流程,必须补齐(或生成)与外部视频一致的 `pose/intrinsics` 与 `latent`(可通过 tokenizer encoder 编码外部视频得到,但效果分布需实测).

## 2026-03-04 13:53 UTC: 为 SDG 推理脚本增加"断点续跑"(按产物进度跳过)

- 覆盖范围:
  - `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py`
  - `cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py`
- 新增 CLI 开关:
  - `--overwrite_existing`: 强制重跑并覆盖已有产物; 默认行为改为"按进度跳过".
- 断点续跑策略(逐类产物):
  - 检查 `pose/`、`intrinsics/`、`latent/`、`rgb/` 是否已存在且可读(对 npz/pkl 做轻量校验).
  - 全部齐全则跳过该输入/该轨迹,避免重复跑 diffusion.
  - 若 `latent` 已存在但 `rgb` 缺失,直接调用 tokenizer 解码补齐 `rgb`,不重新跑 diffusion.
- 资源加载优化:
  - 只有当确实需要生成 latent 时,才初始化分布式(`--num_gpus > 1`)与加载重模型/网络.
  - 仅做 latent 解码时,强制 pipeline `offload_network=True`,避免加载巨大 DiT.
- 自检:
  - `python3 -m compileall` 对两个脚本做了语法编译检查,无报错.

## 2026-03-04 14:25 UTC: 解释 `Ruicheng/moge-vitl` 下载超时与 Warp 初始化告警

- Hugging Face `ReadTimeoutError` 解释:
  - 触发点是 `MoGeModel.from_pretrained("Ruicheng/moge-vitl")` 内部的 `hf_hub_download(..., filename="model.pt")`.
  - 该库默认会先对远端发起 HEAD 获取 ETag/commit,默认超时约 10s,网络慢时会出现重试日志.
  - 本机已存在 MoGe 缓存(约 1.2G),因此该超时更像是"网络慢导致元数据检查超时",而不是"权重不可用".
- Warp `cuDeviceGetUuid` 告警解释:
  - `warp.init()` 初始化 CUDA 后端时尝试获取 `cuDeviceGetUuid` 的 driver entry point,打印告警.
  - 但随后仍能识别 `cuda:0` 并继续加载 `ray_triangle_intersection_warp` kernel,当前不阻断推理流程.
- 运行建议:
  - 若希望减少 HF 噪音或避免卡住: 可设置 `HF_HUB_OFFLINE=1`(确保已缓存)或调大 `HF_HUB_ETAG_TIMEOUT/HF_HUB_DOWNLOAD_TIMEOUT`.
  - 若后续出现 warp kernel 运行期报错,再考虑升级 warp 版本或调整 CUDA driver/toolkit 组合.

## 2026-03-04 15:40 UTC: 修复 `moge-2-vitl` 触发的 MoGe v1/v2 不兼容崩溃

- 问题表现: 推理脚本在加载 `Ruicheng/moge-2-vitl` 时,于 `moge/model/v1.py` 报 `TypeError: getattr(): attribute name must be string`.
- 根因: `moge-2-vitl` 的 checkpoint 属于 MoGe v2 结构,其 `model_config["encoder"]` 为 dict;但脚本固定使用 `moge.model.v1`,导致初始化阶段直接崩溃.
- 修复:
  - `cosmos_predict1/diffusion/inference/inference_utils.py` 新增 `load_moge_model(...)`,按 `model_config["encoder"]` 类型自动选择 `moge.model.v1`/`moge.model.v2`.
  - `gen3c_single_image.py` / `gen3c_single_image_sdg.py` / `gen3c_persistent.py` 统一改为调用 `load_moge_model(...)` 加载 MoGe,不再硬编码 v1.
  - 三个脚本新增 CLI 参数:
    - `--moge_model_id`: 默认 `Ruicheng/moge-2-vitl`,也可切回 `Ruicheng/moge-vitl`.
    - `--moge_checkpoint_path`: 直接使用本地 `model.pt`,用于离线或自定义权重路径.
    - `--hf_local_files_only`: 仅使用 HF 本地缓存,减少网络不稳定导致的 HEAD 超时噪音.
- 验证:
  - `python -m py_compile` 对改动文件做语法检查通过.
  - `load_moge_model(..., hf_local_files_only=True)` 能正确加载 `moge-2-vitl`(v2)与 `moge-vitl`(v1),且模型具备 `.infer(...)`.

## 2026-03-05 00:00 UTC: 对比传统 3DGS pipeline 与 Lyra 方法

- 结论: 传统 3DGS 是 per-scene 迭代优化(多视角 RGB+相机 -> 反复渲染拟合高斯); Lyra 是 feed-forward 3DGS decoder(视频 tokenizer latent + 相机条件 -> 一次前向输出 gaussians),训练依赖相机可控视频扩散模型生成的合成数据做 self-distillation.
- 关键证据:
  - README.md:7-10 直接说明 "Feed-forward" + "video diffusion model self-distillation".
  - gen3c_single_image_sdg.py:765-777/900-906 显示扩散一次采样同时得到 `video` 与 `latents` 并落盘.
  - sample.py:258-269 显示推理优先使用 `rgb_latents` 作为输入.
  - model_latent_recon.py:327-351 显示高斯中心是沿 rays 预测 depth 得到(`pos = rays_os + rays_ds * depths`),属于像素对齐候选高斯.
  - gs.py:18 显示渲染使用 `gsplat.rendering.rasterization`.

## 2026-03-05 09:08 UTC: 优化 `lyra_static_demo_generated` 重影(实验1: 轨迹选择)

- 背景: 实验0(只喂 1 条轨迹)后,用户仍反馈重影很重,说明不只是"多轨迹融合打架".
- 做法: 逐个用 `static_view_indices_fixed=["0".."5"]` 跑 `configs/demo/lyra_static_one.yaml`,并用两类指标对比预测渲染 vs 输入视频:
  - PSNR(越高越好).
  - Laplacian 方差(粗略衡量清晰度,越高通常越锐).
- 结果(同一套 stride=4 的输出设置):
  - view "3"(zoom_out)综合最好(PSNR≈21.61, sharpness≈191).
  - view "4"(zoom_in)次之(sharpness≈186).
  - view "5"(clockwise/orbit)虽然 PSNR 不低,但 sharpness 明显更差(≈94),更像"厚表面重影/糊".
- 落地改动:
  - `configs/demo/lyra_static_one.yaml` 默认轨迹从 `['5']` 调整为 `['3']`.
  - `configs/demo/lyra_static.yaml` 默认轨迹从 `['5']` 调整为 `['3']`.
- 证据截图(便于人工肉眼对比):
  - `outputs/analysis/ghosting_compare/pred_view5_vs_view3_f10.jpg`

## 2026-03-05 09:13 UTC: 让 `sample.py` 支持推理期超参数覆盖(便于做重影对比实验)

- 变更: `sample.py` 允许用 dotlist 在推理期覆盖少量关键超参数(例如 `gaussians_prune_ratio`, `gaussian_scale_cap`, `dnear/dfar`),无需改训练配置文件.
  - 位置: `sample.py` 在加载 `main_config` 后,将这些 key 从推理 config 同步到 `main_config`.
- 快速验证:
  - `python -m py_compile sample.py` 通过.
  - 用 `gaussians_prune_ratio=0.95` 做过一次对比实验,结果 PSNR 明显下降,说明"盲目加大 prune"不是通用解法.

## 2026-03-05 15:38 UTC: 优化 `lyra_static_demo_generated` 重影(实验2: 单轨迹仍重影时的超参数 sweep)

- 背景: 用户补充确认"单 view 也有重影",说明问题不仅是多轨迹融合;实验1 虽然用 view "3" 显著缓解,但仍希望进一步变干净.
- baseline(view "3", 默认参数):
  - PSNR≈21.61
  - sharpness≈191
- sweep 结论(重点是找到"不太掉 PSNR 但更锐"的旋钮):
  - 不建议改 `dfar`: `dfar=50` 会直接跑崩(PSNR≈11.93, sharpness≈2).
  - `gaussian_scale_cap=0.28` 是更稳的 tradeoff:
    - PSNR≈20.92(损失约 0.69)
    - sharpness≈236(提升明显)
    - 对比截图: `outputs/analysis/ghosting_compare/pred_view3_base_vs_scale0p28_f10.jpg`
- 落地改动(让 demo 默认就吃到收益):
  - `configs/demo/lyra_static.yaml` 增加 `gaussian_scale_cap: 0.28`.
  - `configs/demo/lyra_static_one.yaml` 增加 `gaussian_scale_cap: 0.28`.
- 额外验证:
  - 用更新后的 `configs/demo/lyra_static_one.yaml` 跑了一次 cfgcheck,指标与单独覆盖 `gaussian_scale_cap=0.28` 一致(PSNR≈20.92, sharpness≈236).

## 2026-03-05 15:43 UTC: 补充一个更保守的 `gaussian_scale_cap`

- 额外试了 `gaussian_scale_cap=0.29`:
  - PSNR≈21.31(损失约 0.30)
  - sharpness≈212(提升较小)
- 如果更在意保真,可以把 demo config 的 `gaussian_scale_cap` 调回 0.29;如果更在意锐度,0.28 的提升更明显.

## 2026-03-05 16:10 UTC: 拆解方案3 "相机+高斯联合优化"(refinement)为可落地设计与任务清单

- 背景: 用户反馈单轨迹仍然重影很重,希望进入更强的后处理方案.
- 输出内容:
  - `task_plan.md` 追加了新的任务计划: 进入 per-scene refinement,支持 "只优化高斯" 与 "相机+高斯联合" 两种路径.
  - `notes.md` 追加了可落地设计: I/O 契约,两条实现路线(相机可导 vs 通过显式变换高斯规避),参数化,loss/正则,优化日程与风险点.
  - 新增 `specs/joint_refinement_camera_gaussians.md`: 固化流程图/时序图与任务清单,便于后续直接开工实现.


## 2026-03-06 00:25 UTC: 重新分析 `joint_refinement_camera_gaussians` 的推荐顺序

- 重新审视了 `specs/joint_refinement_camera_gaussians.md`,并结合当前代码链路确认: 直接把 `pose_only -> joint` 当默认路线并不稳.
- 新的核心判断:
  - `DeferredBP.backward` 会截断相机梯度,所以当前项目默认 renderer 路径不适合作为直接相机优化入口.
  - 高斯生成是 pixel-aligned(`pos = rays_o + rays_d * depth`),面对时序不一致输入时,更容易形成厚表面/双轮廓.
  - 训练损失缺少 robust masking / residual reweighting,这会推动模型学到“平均化”的糊解.
- 基于现有 demo 产物重新做了 `.ply` 统计:
  - `gaussian_scale_cap=0.28` 的收益主要来自压缩大尺度高斯尾部,不是高斯数量变化.
  - view `5` 的大尺度高斯比例高于 view `3`,与其更严重的重影现象一致.
- 更新后的推荐顺序:
  1. 过滤/重加权高斯 + gaussian-only refinement.
  2. 只在出现整体一致错位证据时再启用 tiny pose-only.
  3. joint refinement 作为最后兜底,不是默认第一步.


## 2026-03-06 00:44 UTC: 学习 `RobustSplat` 后对 Lyra 路线的修正更明确

- 读取了 `https://arxiv.org/abs/2506.02751` 与 `https://github.com/fcyycf/RobustSplat`.
- 核心发现:
  - RobustSplat 的关键不是相机联合优化,而是 `Delayed Gaussian Growth` + `Scale-cascaded Mask Bootstrapping`.
  - 它的方法论非常适合映射到 Lyra 当前的“扩散视频局部不一致导致厚表面重影”问题.
- 对 Lyra 的直接启发:
  - 优先做 residual / feature 驱动的 soft mask 或 weight map.
  - 优先做 staged gaussian-only refinement,延后几何自由度释放.
  - joint camera + gaussians 继续作为最后兜底,不提升优先级.

## 2026-03-06 11:18 UTC: 完成 `Long-LRM style post refinement` 与当前 `refinement_v2` 的 gap review

- 回读并对照了:
  - `specs/long_lrm_style_post_refinement.md`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
  - `src/refinement_v2/` 主线模块
  - `scripts/refine_robust_v2.py`
  - `tests/refinement_v2/` 相关测试
- 结论:
  - 当前 `refinement_v2` 已具备大量可复用能力,并非空骨架.
  - 可直接映射到新 spec 的已有能力包括:
    - `W_robust` 等价实现(`WeightBuilder`)
    - Stage 3A 等价实现(`run_stage2a`)
    - opacity / pruning
    - Stage 3B 等价实现(`run_stage2b`)
    - baseline diagnostics / state / export
- 确认的关键缺口:
  - 入口还不支持 `pose/intrinsics/reference-video` 直连输入.
  - `super_resolved` 当前只是 native GT 的双线性上采样,不是真正外部 SR 视频.
  - patch supervision 已有,但 patch 权重仍是全 1,尚未实现:
    - `gaussian_fidelity_score`
    - `W_sr_select`
    - `W_final_sr`
  - 当前仍缺 `L_sampling_smooth`.
  - Stage 3B gating 还没接到“先跑完 selective SR 再决定”的新语义.
- 产出沉淀:
  - 详细 gap review 已追加到 `notes.md`.
  - `task_plan.md` 已同步记录下一步推荐实现顺序.

## 2026-03-06 06:15 UTC: 将 `Long-LRM style post refinement` spec 拆成 implementation tasks

- 新增计划文件:
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
- 这份计划明确把 `specs/long_lrm_style_post_refinement.md` 拆成了可执行任务序列:
  - CLI / config 对齐
  - `SceneBundle` 与 reference video 对齐
  - baseline render / diagnostics
  - residual / weight map
  - Stage 3A appearance-first
  - opacity / pruning
  - SR patch supervision
  - Stage 3B limited geometry
  - state / resume / export
  - dry-run 与集成验证
- 与旧计划的边界:
  - `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md` 仍保留为更泛化的 V2 路线.
  - 新计划只聚焦 `Long-LRM` 风格的后置 refinement 子路线.
- 上下文续档:
  - 因 `task_plan.md` 与 `notes.md` 已超过 1000 行, 本轮先续档并完成了一次轻量持续学习摘要.
  - 旧长文件已移入 `archive/`, 新档从本轮任务重新开始累积.

## 2026-03-06 07:28 UTC: 正式把 `SplatSuRe` / `Mip-Splatting` / `EDGS` 的结论并入 `Long-LRM style post refinement` 文档体系

- 已更新主规格:
  - `specs/long_lrm_style_post_refinement.md`
- 已同步实现计划:
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
- 本轮写入 spec 的核心变化:
  - `SR patch supervision` 升级为 `SplatSuRe-style selective SR patch supervision`.
  - 新增 `gaussian_fidelity_score`、`W_sr_select`、`W_final_sr = W_robust * W_sr_select`.
  - 将 `Mip-Splatting` 的 `3D smoothing` 吸收为 `L_sampling_smooth`,并明确它不替换原有 `L_scale_ceiling`.
  - 将 `EDGS` 降级为 deferred idea,明确为未来可能的 `EDGS-style local reinitialization`.
- 本轮同步到 plan 的核心变化:
  - Task 4 明确收敛为 `W_robust` 构造.
  - Task 5 明确为 Stage 3A native cleanup.
  - Task 7 升级为 `SplatSuRe-style` selective SR patch supervision,并纳入 fidelity / selection / final SR weight 三个对象.
  - 输出目录布局增加:
    - `gaussian_fidelity_histogram.json`
    - `sr_selection_stats.json`
    - `sr_selection_maps/`
- 文档验证:
  - `beautiful-mermaid-rs --validate-markdown < specs/long_lrm_style_post_refinement.md` 返回 `true`.
  - 更新后的 flowchart 已成功渲染为 Unicode 图,确认图文逻辑一致.


## 2026-03-06 01:05 UTC: 重写 V2 方案并落盘新规格

- 新增 `specs/joint_refinement_camera_gaussians_v2.md`.
- 这版 V2 不是对旧 spec 的小修小补,而是重新定义默认主线:
  - 从“先 pose-only,再 joint”
  - 改成“先鲁棒监督和高斯治理,再视证据决定是否碰 pose”
- V2 规格新增了:
  - 两档实施方案
  - 阶段化流程
  - 决策规则表
  - 重新排序后的任务清单
  - 更贴合 Lyra 的 flowchart 和 sequenceDiagram


## 2026-03-06 01:24 UTC: 学习 `tttLRM` 后对当前 V2 的判断

- 阅读了项目页 `https://cwchenwang.github.io/tttLRM/`, 论文 `https://arxiv.org/abs/2602.20160`, 以及仓库 `https://github.com/cwchenwang/tttLRM` 的 README / `model/model.py` / `model/lact_ttt.py` / 配置文件.
- 结论:
  - `tttLRM` 的核心价值在于长上下文、autoregressive、online progressive 3D reconstruction.
  - 它是 backbone 级别的 LRM 架构升级,不是当前 Lyra 可以低侵入接入的小型后处理.
  - 它没有像 `RobustSplat` 那样直接提供 residual masking / robust weighting 去处理监督不一致.
- 对当前任务的影响:
  - 不改变 `joint_refinement_camera_gaussians_v2` 的主线.
  - 适合作为后续“远期架构升级路线”的参考,而不是当前 ghosting 修复的一线方案.


## 2026-03-06 04:49 UTC: 评估 `LongSplat` 对当前 Lyra 重影问题的帮助边界

- 阅读了 `LongSplat` 的项目页、论文摘要以及仓库关键实现(`README.md`、`train.py`、`utils/pose_utils.py`、`utils/graphics_utils.py`、`arguments/__init__.py`).
- 结论:
  - `LongSplat` 的主问题是 `unposed casual long videos` 的增量式 3DGS 重建.
  - 它的核心能力落在 `PnP/pose refinement + local/global optimization + adaptive anchor/octree + long-video scalability`.
  - 它不是面向当前 Lyra 单轨迹 ghosting 的直接方案,也不应替代 `joint_refinement_camera_gaussians_v2` 的主线.
- 对当前任务的影响:
  - 不改变现有 V2 的优先级.
  - 仍保持 `RobustSplat -> residual weighting -> gaussian-only refinement -> tiny pose-only -> joint fallback` 这条主线.
  - 将 `LongSplat` 归类为后续 "无位姿长视频 / 多段轨迹 / pose drift / 增量融合" 场景下的重要参考路线.


## 2026-03-06 05:03 UTC: 将 `LongSplat` 正式补入 `joint_refinement_camera_gaussians_v2` 的未来路线

- 更新 `specs/joint_refinement_camera_gaussians_v2.md`,新增 `Future / V3: LongSplat-style 无位姿长视频增量重建`.
- 文档中正式固定了 `LongSplat` 的定位:
  - 不进入当前 ghosting 修复主线.
  - 作为后续 `无 pose 长视频 / 多段轨迹 / 增量融合 / pose drift` 场景下的架构升级路线.
- 同时明确了长期组合关系:
  - `V3` 负责新帧注册、滑窗优化、长程组织.
  - `V2` 负责 residual weighting、gaussian-only refinement 与去重影.


## 2026-03-06 05:18 UTC: 深化 `joint_refinement_camera_gaussians_v2` 到工程可执行级别

- 更新 `specs/joint_refinement_camera_gaussians_v2.md`,补齐了 V2 的执行细节.
- 本轮新增内容包括:
  - 阶段门控与回退规则
  - `weight_map` 构造细节与默认映射方式
  - 参数分组与各阶段可训练变量
  - 第一版默认超参数建议表
  - `diagnostics.json` / `metrics_stage*.json` 的最小输出结构
  - 失败模式与回退手册
  - 工程模块拆分建议
  - 验证协议与默认通过线
- 这使得 V2 从“方向正确”进一步推进到“可以直接拆脚本和模块开工”的状态.


## 2026-03-06 05:37 UTC: 探索“超分视频是否能直接改善 `sample.py` 高斯清晰度”

- 回读并核对了 `sample.py`、`src/models/data/provider.py`、`src/models/data/radym.py`、`src/models/utils/data.py`、`src/models/utils/model.py`、`src/rendering/gs_deferred.py` 以及当前 demo / training 配置.
- 确认了当前主链约束:
  - 默认 demo 配置会优先读取 `latent/*.pkl`,不是优先依据 mp4 重新编码.
  - provider 会把 RGB 与 intrinsics 一起裁剪缩放到 `img_size`.
  - 当前主档位固定在 `img_size: [704, 1280]`.
- 由此得出的判断:
  - 单纯把上游视频换成更高分辨率版本,并不会自然变成“更清晰的高斯输入”.
  - 这条路若硬做,已经接近“新分辨率支持工程”,不是简单替换视频.
- 同时确认了更可行的后处理路线:
  - `sample.py` 输出高斯后,再接一个独立的 scene-level refinement 阶段更合理.
  - 当前 deferred renderer 会保留高斯梯度,但截断相机梯度,因此首选 `gaussian-only refinement`.
  - 若使用超分后的对等视频做监督,需要保证内容/时序完全对齐,并按倍率同步缩放 intrinsics.
- 最终建议:
  - 不推荐“先超分,再直接回灌 `sample.py` 主链”.
  - 推荐“`sample.py` 作为初始化器 + 超分对等视频驱动的 post-refinement”.


## 2026-03-06 06:08 UTC: 评估 `tttLRM` 与 `Long-LRM` 对 Lyra 的借鉴价值

- 阅读并对照了:
  - `tttLRM` 项目页 / GitHub / arXiv
  - `Long-LRM` 项目页 / GitHub / arXiv
- 结论不是“二选一谁更强”,而是它们适合 Lyra 的层次不同:
  - `Long-LRM` 更适合当前 Lyra,因为它的主问题就是“长序列 feed-forward GS 重建”.
  - `tttLRM` 更适合中长期研究,因为它的核心创新在 `TTT / LaCT fast-weight memory + streaming online update`.
- 对当前 Lyra 最可直接借用的部分:
  - `Long-LRM` 的 `post-prediction optimization`
  - `Long-LRM` 的 opacity regularization / pruning 思路
  - `Long-LRM` 的长序列骨干与 token budget 管理方法论
- 对当前 Lyra 不建议直接照搬的部分:
  - `tttLRM` 的主干与状态更新机制,因为这已经属于“下一代重建器范式”,不是给现有链路补一层小改动.
- 代码层判断:
  - `Long-LRM` 的 self-reimplemented repo 更适合拿来学习工程组织、训练/推理入口和后优化结构.
  - `tttLRM` 当前更适合学方法和推理组织,因为 README 已明确训练代码并未完整提供.
- 最终排序建议:
  - 现在推进 Lyra 当前路线时,优先参考 `Long-LRM`.
  - 如果后续转向长视频增量融合 / 在线状态更新 / 新主干研究,再重点下探 `tttLRM`.


## 2026-03-06 06:22 UTC: 输出 `Long-LRM` / `tttLRM` 到 Lyra 的模块迁移清单

- 继续把外部项目分析从“方向判断”推进到“文件级映射”.
- 关键新结论:
  - Lyra 其实已经内含一部分 `Long-LRM` 风格设计:
    - `Mamba2 + Transformer` 混合骨干
    - gaussian pruning
    - opacity regularization
    - token / gaussian 子采样
  - 因此最值得借的不是“整套新主干”,而是把这些已有地基进一步系统化.
- 当前最值得推进的 3 个可迁移模块:
  1. `Long-LRM` 的 post-prediction optimization
  2. `Long-LRM` 的 opacity / pruning / diagnostics 体系
  3. `Long-LRM` 的 token budget 管理思路(但优先级低于前两者)
- 中侵入但暂不优先的方向:
  - 更系统的 token merging
  - 受限 geometry offset / schedule
  - 多视图 / 长序列 curriculum
- 高侵入且不适合当前立即做的方向:
  - `tttLRM` 的 LaCT / fast-weight memory
  - streaming autoregressive reconstruction
  - sequence parallel + online state update 主干
- 最终落地顺序建议:
  - 先把 `refine_robust_v2` 的最小闭环做出来
  - 在这个闭环里优先吸收 `Long-LRM`
  - 等这条线稳了,再考虑 `tttLRM` 级别的新范式


## 2026-03-06 06:42 UTC: 新增 `Long-LRM` 风格的后置 refinement 规格

- 按当前探索结论,新增了聚焦版规格:
  - `specs/long_lrm_style_post_refinement.md`
- 这份规格不再泛谈 V2 总原则,而是把方案明确收敛为:
  - `sample.py` 负责 feed-forward 初始化
  - `post-refinement` 负责短迭代高斯优化
  - 超分后的对等视频只在这个后置阶段作为可选监督信号
- 文档内容已覆盖:
  - 为什么不该把超分视频直接回灌主链
  - 为什么该借 `Long-LRM` 的 post-prediction optimization
  - 输入输出契约
  - patch-based 高分辨率监督策略
  - Phase / Stage 拆分
  - 仓库落点与 CLI 草案
- 文档中的两个 mermaid 图块已使用 `beautiful-mermaid-rs --ascii` 成功渲染,确认语法可用.
- 这使得当前路线从“方向判断”进一步推进到“可直接据此开工实现”的规格状态.


## 2026-03-06 05:27 UTC: 继续深化 `joint_refinement_camera_gaussians_v2` 的 CLI 与接口层设计

- 更新 `specs/joint_refinement_camera_gaussians_v2.md`,新增 V2 的 CLI、目录布局、配置对象、模块接口和状态机设计.
- 本轮主要固定了:
  - `scripts/refine_robust_v2.py` 的参数草案
  - 断点续跑所需的 `state/` 目录与最小恢复约定
  - `RefinementRunConfig` / `StageHyperParams` / `SceneBundle` 等对象层次
  - `stage_controller` / `weight_builder` / `gaussian_adapter` / `diagnostics` 的职责边界
  - 推荐的实现顺序,明确 `joint fallback` 必须最后进入
- 这使得 V2 已经接近“可以直接按规格开始搭脚手架实现”的状态.


## 2026-03-06 05:36 UTC: 为 `joint_refinement_camera_gaussians_v2` 补充文件级实现蓝图与主流程伪代码

- 更新 `specs/joint_refinement_camera_gaussians_v2.md`,新增文件级职责拆分、主流程伪代码、测试矩阵和实现陷阱说明.
- 本轮进一步固定了:
  - 哪些逻辑放在脚本层,哪些逻辑必须沉到模块层
  - `main()` / `RefinementRunner.run()` / `run_stage2a()` 的推荐控制流
  - 第一版最小测试矩阵
  - 需要优先规避的工程陷阱
- 这使得 V2 已经可以直接转换成“按文件拆任务”的实现计划.


## 2026-03-06 05:48 UTC: 将 `joint_refinement_camera_gaussians_v2` 转成文件级 implementation task list

- 新增 `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md`.
- 该文档把 V2 规格进一步转换为可执行的任务清单,覆盖:
  - 入口脚本与模块骨架
  - 配置层 / 数据层 / 高斯适配器 / 权重构造器 / 阶段控制器 / 诊断 / 状态恢复
  - `Phase 0`、`Stage 2A`、`Stage 2B`、`Phase 3`、`Phase 4` 的实现顺序
  - 对应测试文件与运行命令
  - MVP 与完整版本的优先级划分


## 2026-03-06 06:24 UTC: 完成 `joint_refinement_camera_gaussians_v2` 的 Task 1 脚手架搭建

- 新建 `scripts/refine_robust_v2.py`,作为后续 refinement 的唯一入口脚本.
- 新建 `src/refinement_v2/` 包及其骨架模块:
  - `config.py`
  - `data_loader.py`
  - `runner.py`
  - `stage_controller.py`
  - `weight_builder.py`
  - `gaussian_adapter.py`
  - `losses.py`
  - `diagnostics.py`
  - `state_io.py`
- 新建 `tests/refinement_v2/__init__.py`,为后续测试落点预留目录.
- 验证:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py` 通过.

## 2026-03-06 07:28 UTC: 完成 `joint_refinement_camera_gaussians_v2` 第一版可运行落地与真实 subset 验证

- 已完成 `refinement_v2` 主线实现:
  - `scripts/refine_robust_v2.py`
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/data_loader.py`
  - `src/refinement_v2/gaussian_adapter.py`
  - `src/refinement_v2/weight_builder.py`
  - `src/refinement_v2/losses.py`
  - `src/refinement_v2/diagnostics.py`
  - `src/refinement_v2/stage_controller.py`
  - `src/refinement_v2/state_io.py`
  - `src/refinement_v2/runner.py`
- 已完成 `tests/refinement_v2/` 的单测与轻集成测试,最终结果:
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - `25 passed in 3.83s`
- 真实验证时补齐了三类关键兼容性修复:
  - `WeightBuilder` 的 EMA 权重图改为 detached,避免跨 iter 计算图复用崩溃.
  - `data_loader` 改为正确合并 demo YAML 的 `config_path` 链.
  - refinement CLI 新增 `--dataset-name`,且 refinement loader 会关闭 `use_depth/load_latents`,并补 `target_index_manual=None`.
- 已完成真实产物验证:
  - `outputs/refine_v2/view3_phase0_subset`
  - `outputs/refine_v2/view3_stage2a_subset`
  - `outputs/refine_v2/view5_stage2a_subset`
- 关键结果:
  - view `3` subset, `Stage 2A` 5 iter: `PSNR 19.2813 -> 21.9934`, `residual_mean 0.066586 -> 0.043972`.
  - view `5` subset, `Stage 2A` 5 iter: `PSNR 19.0454 -> 24.2871`, `residual_mean 0.068201 -> 0.034461`.
- 当前结论:
  - `Phase 0 + Phase 1 + Stage 2A` 已经不是纸面规格,而是可在现有 demo 资产上真实运行的第一版.
  - `Stage 2B / Phase 3 / Phase 4` 代码路径与测试已具备,但更适合下一轮围绕更长迭代和更强监督继续打磨.

## 2026-03-06 07:48 UTC: 为 `refinement_v2` 增加 before/after 渲染导出并完成全序列真实验证

- 新增可视化导出能力:
  - `src/refinement_v2/diagnostics.py` 现在支持导出 `baseline_render.mp4`、`final_render.mp4`、`gt_reference.mp4` 以及首帧 PNG.
  - `src/refinement_v2/runner.py` 已在 `Phase 0` 和最终导出阶段自动调用这些可视化导出.
  - `diagnostics.json` 现在会包含 `artifacts` 路径映射.
- 兼容性处理:
  - 当前系统 `python3` 没有 `imageio`,因此实现改为:
    - 优先 `imageio`
    - 否则回退系统 `ffmpeg`
    - `ffmpeg` 再按 `libx264 -> mpeg4` 顺序尝试编码器
- 测试补强:
  - `tests/refinement_v2/test_diagnostics.py`
  - `tests/refinement_v2/test_runner_phase0.py`
  - `tests/refinement_v2/test_runner_stage2a.py`
- 最终回归结果:
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - `26 passed in 4.28s`
- 已完成完整 target 序列的真实验证:
  - `outputs/refine_v2/view3_stage2a_full`
  - `outputs/refine_v2/view5_stage2a_full`
- 关键结果:
  - view `3` 全序列: `PSNR 19.6426 -> 22.9245`, `residual_mean 0.063627 -> 0.039699`.
  - view `5` 全序列: `PSNR 18.7739 -> 21.7612`, `residual_mean 0.072603 -> 0.047303`.
- 当前结论:
  - 可视化导出已经进入正式主线,现在可以直接打开 mp4 做肉眼对比.
  - `Stage 2A` 在完整序列上依然有效,但对 sharpness 的帮助不稳定,下一步应优先往 `opacity/pruning` 与 `patch supervision` 继续推进.

## 2026-03-06 08:33 UTC: 完成 `refinement_v2` 的 opacity/pruning 落地与真实验证

- 已完成 pruning 主线代码落地:
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/stage_controller.py`
  - `src/refinement_v2/gaussian_adapter.py`
  - `src/refinement_v2/diagnostics.py`
  - `src/refinement_v2/runner.py`
- 已完成 pruning 测试补强:
  - `tests/refinement_v2/test_config.py`
  - `tests/refinement_v2/test_stage_controller.py`
  - `tests/refinement_v2/test_diagnostics.py`
  - `tests/refinement_v2/test_pruning.py`
- 新增能力包括:
  - CLI/配置层 pruning 开关与阈值、频率、warmup、比例上限、最小保留数
  - `StageController.should_prune_now()`
  - `GaussianAdapter.collect_prune_candidates()`
  - `GaussianAdapter.prune_low_opacity()`
  - `DiagnosticsWriter.write_prune_summary()`
  - `runner.run_stage2a()` 内 step 后 pruning + optimizer 重建
- 回归验证:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py` 通过
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 通过, 共 `34 passed`
- 真实验证产物:
  - `outputs/refine_v2/view5_stage2a_prune_full`
  - `outputs/refine_v2/view3_stage2a_prune_full`
- 关键真实结果:
  - view `5` 相比无 pruning full run:
    - `PSNR 21.7612 -> 21.8330`
    - `residual_mean 0.047303 -> 0.047021`
    - `sharpness 0.00162135 -> 0.00164230`
    - `opacity_lowconf_ratio 0.608446 -> 0.599079`
    - `num_gaussians 340735 -> 314286`
  - view `3` 相比无 pruning full run:
    - `PSNR 22.9245 -> 22.9937`
    - `residual_mean 0.039699 -> 0.039381`
    - `sharpness 0.00232026 -> 0.00240403`
    - `opacity_lowconf_ratio 0.496800 -> 0.487772`
    - `num_gaussians 340735 -> 314286`
- 当前结论:
  - pruning 已经从规格变成可执行主线.
  - 它能稳定减少低置信高斯和总高斯数量,并带来小幅但真实的质量收益.
  - 这一步对“去雾状叠层 / 降低低价值高斯堆积”是有效的,但还不是最终质量上限.

## 2026-03-06 08:39 UTC: 进一步减轻 pruning 诊断摘要体积

- `src/refinement_v2/gaussian_adapter.py` 的 pruning 摘要已从完整 `pruned_indices` 改为 `pruned_indices_preview`.
- 这样真实大场景下不会把几千个索引直接塞进 `pruning_summary.json`.
- 已重新执行 `view5_stage2a_prune_full` 与 `view3_stage2a_prune_full`,确认:
  - 关键指标保持一致
  - `pruning_summary.json` 体积约 `4.0K`

## 2026-03-06 09:12 UTC: 完成 `refinement_v2` 的 patch-based supervision 第一版落地

- 已完成 patch supervision 主线代码落地:
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/data_loader.py`
  - `src/refinement_v2/losses.py`
  - `src/refinement_v2/runner.py`
- 已完成 patch supervision 测试补强:
  - `tests/refinement_v2/test_config.py`
  - `tests/refinement_v2/test_data_loader.py`
  - `tests/refinement_v2/test_patch_supervision.py`
  - `tests/refinement_v2/helpers.py`
- 新增能力包括:
  - `reference_mode/sr_scale/patch_size` 配置映射
  - `SceneBundle.reference_images/intrinsics_ref` 数据契约
  - native -> reference patch window 坐标映射
  - patch intrinsics 主点偏移
  - `stage2a` 内的 `loss_patch_rgb/loss_patch_perceptual`
- 回归验证:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py` 通过
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 通过, 共 `41 passed`
- 真实验证产物:
  - `outputs/refine_v2/view5_stage2a_prune_patch_full`
  - `outputs/refine_v2/view5_stage2a_prune_patch_full_l025`
  - `outputs/refine_v2/view3_stage2a_prune_patch_full`
- 关键真实结果:
  - view `5`, `lambda_patch_rgb=0.5`:
    - `PSNR 21.8330 -> 21.8592`
    - `residual_mean 0.047021 -> 0.047053`
  - view `5`, `lambda_patch_rgb=0.25`:
    - `PSNR 21.8330 -> 21.8907`
    - `residual_mean 0.047021 -> 0.046883`
  - view `3`, `lambda_patch_rgb=0.5`:
    - `PSNR 22.9937 -> 23.0862`
    - `residual_mean 0.039381 -> 0.039112`
- 当前结论:
  - patch supervision 已经从“待办”变成可执行主线.
  - 它能继续小幅压低局部误差,但当前主要收益仍然体现在对齐而不是锐度恢复.
  - 困难轨迹上,更保守的 patch 权重(`0.25`)比 `0.5` 更稳.

## 2026-03-06 10:37 UTC: 完成 `refinement_v2` 的 `Stage 2B limited geometry` 落地与真实验证

- 已完成 `Stage 2B` 主线代码落地:
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/losses.py`
  - `src/refinement_v2/gaussian_adapter.py`
  - `src/refinement_v2/stage_controller.py`
  - `src/refinement_v2/runner.py`
- 已完成测试补强:
  - `tests/refinement_v2/test_stage_controller.py`
  - `tests/refinement_v2/test_runner_stage2b.py`
  - `tests/refinement_v2/test_config.py`
- 这轮新增能力包括:
  - `lambda_means_anchor` / `lambda_rotation_reg` 的配置与 CLI 映射
  - `compute_means_anchor_loss()`
  - `compute_rotation_regularization_loss()`
  - `GaussianAdapter.initial_rotations` buffer,并在 pruning 后同步维护
  - 更保守的 `should_enter_stage2b()` 门控
  - `Stage 2B` 内继续复用 patch supervision
  - `Stage 2B` metrics 中新增 `loss_means_anchor` / `loss_rotation_reg`
- 回归验证:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py` 通过
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 通过,共 `47 passed`
- 真实验证过程中还确认了环境事实:
  - 当前系统 `py310` 环境缺 `flash_attn`
  - 真实资产验证需要用 `PYTHONPATH="$(pwd)" pixi run python ...`
- 真实验证产物:
  - `outputs/refine_v2/view5_stage2b_from_stage2a_patch_l025`
- 关键真实结果(以该 run 的 Phase 0 baseline,也就是已验证 `Stage 2A` 基线为对照):
  - `PSNR 21.9213 -> 22.5822` (`+0.6609`)
  - `residual_mean 0.04648 -> 0.04165`
  - `sharpness 0.001587 -> 0.001634`
  - `scale_tail_ratio 0.014566 -> 0.014360`
  - `opacity_lowconf_ratio 0.596724 -> 0.585976`
- 当前结论:
  - `Stage 2B` 已经不是占位阶段,而是可执行且在真实 `view 5` 上有正收益的 limited geometry 步骤.
  - 进入门控原先过保守,已根据真实 `view 5` 结果把 `local_overlap_persistent` 阈值从 `0.05` 调整为 `0.045`.
  - 当前更推荐把 `Stage 2B` 用在“已有较稳 Stage 2A 基线之后”的增量优化,而不是一开始就抢跑.

## 2026-03-06 11:18 UTC: `Long-LRM style post refinement` gap review 尾部收口

- 本轮补做了新 spec 与当前 `refinement_v2` 代码现实的逐项对照.
- 最终确认:
  - 已有主体:
    - `W_robust` 等价实现
    - Stage 3A 等价实现
    - opacity/pruning
    - Stage 3B 等价实现
    - baseline / state / export
  - 主要缺口:
    - `pose/intrinsics/reference-video` 直连输入
    - 外部 SR/reference 视频接入
    - `gaussian_fidelity_score`
    - `W_sr_select`
    - `W_final_sr`
    - `L_sampling_smooth`
    - `Phase 3S / Stage 3SR` 显式阶段
- 后续实现优先级也已重排:
  1. direct file inputs + external reference contract
  2. selective SR 阶段拆分
  3. fidelity / selection / smoothing
  4. Stage 3B gating 对齐新主线

## 2026-03-06 11:32 UTC: 复核 `Long-LRM` 与 `joint_v2` 的任务边界

- 根据用户提醒,重新核对了当前任务边界。
- 现在明确:
  - `specs/long_lrm_style_post_refinement.md`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
  才是本线程要服务的目标文档。
- `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md` 即使正在被另一进程实现,也只能作为:
  - 共享代码现实
  - 可复用实现来源
  不能直接作为本线程任务完成状态的依据。
- 因此本线程后续的正确口径是:
  - 可以引用共享代码里已经出现的能力,说明 `Long-LRM` 路线未来可复用什么。
  - 但不会再把这些能力直接表述成 `Long-LRM` 计划任务已经完成。

## 2026-03-06 11:02 UTC: 为 `Stage 2B` 增加显式 `start_stage` workflow

- 已完成 `Stage 2B` warm-start workflow 的代码落地:
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/runner.py`
- 已完成测试补强:
  - `tests/refinement_v2/test_config.py`
  - `tests/refinement_v2/test_runner_stage2b.py`
- 这轮新增能力包括:
  - CLI 新增 `--start-stage {stage2a,stage2b}`
  - `start_stage=stage2b` 时,会保留 `Phase 0 + Phase 1`,但跳过新的 `Stage 2A` optimizer step
  - `RefinementRunner.bootstrap_stage2b_from_current_gaussians()`
  - `diagnostics.json` 新增:
    - `start_stage`
    - `warm_start_stage2b`
- 回归验证:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py` 通过
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 通过,共 `48 passed`
- 真实验证产物:
  - `outputs/refine_v2/view5_stage2b_startstage_cli_l025`
- 关键真实结果(相对该 run 的 Phase 0 baseline,也就是输入的 `gaussians_stage2a.ply`):
  - `PSNR 21.9213 -> 22.4290`
  - `residual_mean 0.04648 -> 0.04247`
  - `sharpness 0.001587 -> 0.001631`
- 与旧的手工 warm-start 结果相比:
  - 新 CLI 略低于 `view5_stage2b_from_stage2a_patch_l025`
  - 原因更像旧方案里额外多跑了 1 轮 `Stage 2A`,而不是新 CLI workflow 有问题
- 当前结论:
  - 现在已经不需要再靠“手工指定 `gaussians_stage2a.ply` + 额外绕一次 Stage 2A”来进入 `Stage 2B`
  - `--start-stage stage2b` 已经成为明确、可复用、可验证的正式 workflow

## 2026-03-06 11:36 UTC: 为 `refinement_v2` 接入 external reference 输入契约

- 已完成 external reference 主线代码落地:
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/data_loader.py`
- 已完成测试补强:
  - `tests/refinement_v2/test_config.py`
  - `tests/refinement_v2/test_data_loader.py`
- 这轮新增能力包括:
  - CLI 新增:
    - `--reference-path`
    - `--reference-intrinsics-path`
  - `reference_path` 现支持两类输入:
    - 外部帧目录
    - 本地视频文件(`mp4` 等)
  - `data_loader` 新增 external reference 对齐逻辑:
    - 允许“完整时序 reference”按 `frame_indices` 再裁一遍
    - 支持从 `reference_intrinsics_path` 的 `npz` 覆盖 `intrinsics_ref`
    - 若未提供 external intrinsics,则在 `super_resolved` 模式下按外部分辨率自动推断并缩放 native intrinsics
- 回归验证:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py` 通过
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 通过,共 `51 passed`
- 真实验证1: 直接探测 external 2x mp4 是否进入 `SceneBundle`
  - 外部 reference: `outputs/refine_v2/reference_view5_rgb_2x.mp4`
  - 探测结果:
    - `gt_images = (1, 3, 3, 704, 1280)`
    - `reference_images = (1, 3, 3, 1408, 2560)`
    - `sr_scale = 2.0`
    - `intrinsics_ref` 已正确变成 native intrinsics 的 2 倍
- 真实验证2: 用 external 2x mp4 跑一轮最小 Stage 2A
  - 产物:
    - `outputs/refine_v2/view5_external_reference_2x_subset`
  - 关键证据:
    - run 正常停在 `stage2a`
    - `metrics_stage2a.json` 中出现:
      - `loss_patch_rgb = 0.364529...`
      - `loss_patch_perceptual = 0.364529...`
- 当前结论:
  - `reference_images` 现在已经不再只依赖 native GT 的内部上采样占位.
  - external reference 视频/帧序列已经能真实进入 patch supervision 主线.
  - 当前还没做的是“直接从 pose/intrinsics/rgb 文件绕过 dataloader 构建完整 scene bundle”,以及 selective SR 的 fidelity / selection / smoothing 三件套.

## 2026-03-06 11:44 UTC: 将 `joint_refinement_camera_gaussians_v2` 计划文档同步为最终完成版

- 已更新 `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md`。
- 本次同步不是补功能代码,而是把文档状态从“施工期计划”升级成“最终完成版”。
- 主要更新包括:
  - 顶部新增 `Final Status` / `Completion Summary`
  - 明确 `Task 1 ~ Task 13` 已全部完成
  - 保留原任务清单作为历史执行记录
  - 新增后续增量说明:
    - `--start-stage stage2b`
    - external reference contract(`--reference-path` / `--reference-intrinsics-path`)
  - `Implementation Notes` 补入最新真实验证结论与当前边界
- 当前效果:
  - 现在这份文档既能作为“已完成实现”的总览,又不会丢失原始 task-by-task 历史上下文。

## 2026-03-06 11:45 UTC: 基于最新代码现实重排 `Long-LRM` 后续路线

- 重新核对了 `Long-LRM` 自身 spec/plan 与最新共享代码现实。
- 当前结论发生了一个关键变化:
  - external reference contract 已经进入共享代码主线,不再是 `Long-LRM` 线的头号阻塞。
- 因此 `Long-LRM` 这条线接下来的真正优先级改为:
  1. 把现有 patch supervision 从 Stage 3A 中显式拆成:
     - Stage 3A native cleanup
     - Phase 3S fidelity / selection
     - Stage 3SR selective SR
  2. 把 `gs.py` 里 `rasterization(...).info` 往 refinement 层抬,作为 `gaussian_fidelity_score` 的统计入口。
  3. 落地:
     - `gaussian_fidelity_score`
     - `W_sr_select`
     - `W_final_sr`
  4. 再补 `L_sampling_smooth` 作为 selective SR 后的保护性约束。
- 同时明确:
  - Stage 3B 的共享代码主体虽然存在,但对 `Long-LRM` 线来说,它现在不是第一优先级。
  - 更准确的定位是: 等前面的 selective SR 阶段成型后,再把 Stage 3B gating 接过去。

## 2026-03-06 11:52 UTC: 记录并复核 `SplatSuRe + Mip-Splatting` 在 `Long-LRM` 方案中的定位

- 已把用户确认的方案约束正式追加到四文件上下文。
- 复核结果:
  - `SplatSuRe` 的核心价值已明确写入方案:
    - per-Gaussian fidelity score
    - per-view SR weight map
    - LR + SR selective joint objective
  - 其顺序约束也已明确写入:
    - 先 Stage 3A native cleanup
    - 再上 selective SR
  - `Mip-Splatting` 的定位也已明确:
    - 当前主要吸收 3D smoothing / sampling-frequency constraint 思想
    - 用于约束 HR supervision 下的不受支持高频与 aliasing
    - 不作为厚表面 / 双轮廓 / SR 假细节一致性的主解法
  - 方案中已经明确:
    - `L_sampling_smooth` 不是 `L_scale_ceiling` / 旧 `scale_reg` 的替代
    - 而是第二条互补约束
- 当前结论:
  - 从 spec / implementation plan 角度,这些判断已经被考虑并写进方案。
  - 从代码实现角度,`SplatSuRe-style selective SR` 与 `Mip-inspired smoothing` 仍属于下一阶段待落地核心。

## 2026-03-06 11:58 UTC: 将 `joint_refinement_camera_gaussians_v2` 主规格同步到当前实现状态

- 已更新 `specs/joint_refinement_camera_gaussians_v2.md`。
- 本轮重点不是新增代码功能,而是让主规格和当前代码现实重新对齐。
- 主要更新包括:
  - 顶部新增“当前实现状态(2026-03-06)”
  - 数据契约补入:
    - `dataset_name`
    - `start_stage`
    - `reference_mode/sr_scale`
    - `reference_path`
    - `reference_intrinsics_path`
  - `diagnostics.json` 示例补入:
    - `start_stage`
    - `warm_start_stage2b`
  - CLI 设计草案补入 external reference 参数
  - `RefinementRunConfig` / `StageHyperParams` / `SceneBundle` 草案同步到当前实际字段
  - 任务清单从未完成改为全部已完成
  - 新增“已完成验证摘要(2026-03-06)”收口真实验证结果
- 当前效果:
  - 现在主规格、计划文档和代码状态三者已经重新对齐。

## 2026-03-06 12:18 UTC: 说明 `joint_refinement_camera_gaussians_v2` 的实际使用方式

- 已核对 `README.md` 的 Example 1 与当前 `refine_robust_v2` 实现入口。
- 结论:
  - README 的两条命令只覆盖“SDG 生成 + baseline 3DGS 重建”。
  - 若要使用 `joint_refinement_camera_gaussians_v2` 增强能力,还需要额外执行第 3 步 `scripts/refine_robust_v2.py`。
- 已确认关键衔接点:
  - `sample.py` 会把 baseline 渲染写到 `main_gaussians_renderings/`。
  - `sample.py` 在 `save_gaussians_orig=true` 时会导出 `gaussians_orig/gaussians_0.ply`,供 refinement 直接载入。
  - `refine_robust_v2.py` 已支持 `--reference-path`、`--reference-intrinsics-path`、`--start-stage stage2b` 等增强入口。
- 已发现当前仓库中的实际配置差异:
  - `configs/demo/lyra_static.yaml` 现在不是 README 里的默认状态。
  - 当前 `dataset_name` 为 `lyra_static_demo_generated_one`,且主视角固定为 `3`。
  - 因此若用户想消费 `assets/demo/static/diffusion_output_generated`,建议显式用 CLI override `dataset_name=lyra_static_demo_generated`,不要机械照抄 README 文案。

## 2026-03-06 12:27 UTC: 新增独立使用说明文档

- 已创建独立使用文档:
  - `docs/joint_refinement_camera_gaussians_v2_usage.md`
- 这份文档面向“如何实际使用已实现的 v2 增强部分”这个问题。
- 文档当前覆盖:
  - README 两步为什么只属于 baseline
  - 为什么还必须单独跑第 3 步 `scripts/refine_robust_v2.py`
  - baseline `.ply` 该接哪一个文件
  - 当前仓库 `configs/demo/lyra_static.yaml` 与 README 的配置差异
  - 三种实用用法:
    1. 内置 demo
    2. `diffusion_output_generated`
    3. external SR reference
  - `--start-stage stage2b` 的适用场景
- 已复核:
  - 命令参数名与当前 CLI 一致
  - 示例路径与当前仓库真实输出结构一致
  - 标题层级已整理,方便后续继续补充

## 2026-03-06 12:36 UTC: 完成 git 提交并推送

- 已完成提交前新鲜验证:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py`
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - 结果: `51 passed`
- 已创建提交:
  - `84d7a36 Implement refinement v2 stage2b and reference workflow`
- 已推送到远端:
  - `origin/main`
  - 推送结果: `fd8f963..84d7a36  main -> main`
- 本次推送包含:
  - `refinement_v2` 的 `stage2b` / external reference 相关实现与测试
  - `joint_refinement_camera_gaussians_v2` 规格与计划文档同步
  - 独立使用说明文档 `docs/joint_refinement_camera_gaussians_v2_usage.md`

## 2026-03-08 07:04 UTC: 在 A800 上建立 `target_subsample=4` 的 full-view native `stage2a` 容量边界

### 任务内容
- 在新切换的 A800 主机上, 复用当前 full-view native `stage2a` 口径, 建立 `target_subsample=4` 的基准。
- 目标不是直接改代码, 而是先确认这一档 observation 密度在当前实现上到底“能跑”还是“直接撞墙”。

### 完成过程
- 先回读 `task_plan.md`、`WORKLOG.md`、`notes.md`、`LATER_PLANS.md`、`ERRORFIX.md`, 补齐上一轮 `Long-LRM` selective SR 主线的上下文。
- 确认当前文档里的 full-view 正式基线仍停留在 `target_subsample=16`, 并验证 A800 设备为 `NVIDIA A800-SXM4-80GB`。
- 对已有 `full_view_native_stage2a_fair_v2/v3` 产物做复核:
  - baseline/final 视频仍为 `48` 帧
  - 说明它们与 `sub16` 同档, 不是 `sub4`
- 执行了两轮 `sub4` 最小 smoke:
  1. 默认 allocator
  2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 两轮都在 `Phase 0` 的 [`gs.py`](/workspace/lyra/src/rendering/gs.py) `render_meta` 合并路径 OOM, 未能进入 `stage2a` 正式迭代。

### 总结感悟
- 这轮已经把 `sub4` 的 benchmark 建立成了明确的“容量边界”而不是“可运行基线”。
- 当前真正限制更高 observation 密度的, 不是 `stage2a` loss 或 patch supervision, 而是 renderer 把 dense per-view meta 全量合并回 `[B, V, ...]` 的内存策略。
- 后续如果还想冲 `sub4`, 最值得优先动的不是再换机器, 而是先削减或延迟 `render_meta` 的物化方式。

## [2026-03-09 04:20:00] 任务名称: 再次分析 add-refinement-v2-depth-anchor 可行性

### 任务内容
- 回读六文件上下文与 OpenSpec change `add-refinement-v2-depth-anchor`
- 对照 `refinement_v2` 当前实现,重新评估 depth anchor 的工程可行性与价值边界
- 补充静态证据、动态验证结果与后续建议

### 完成过程
- 阅读了 `proposal.md`、`design.md`、`tasks.md` 与 `spec.md`,确认 change 主张是用 baseline render depth 做 self-anchor。
- 对照了 `runner.py`、`gaussian_adapter.py`、`losses.py`、`data_loader.py`、`gs.py`、训练期 `src/models/utils/loss.py` 的 depth 归一化逻辑。
- 运行了相关回归测试:
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_runner_phase0.py tests/refinement_v2/test_runner_stage2a.py tests/refinement_v2/test_patch_supervision.py`
  - 结果 `18 passed`
- 额外用最小脚本确认 `Stage 2A` 会改 `opacity/colors`,不会改 `means`,从而验证了“appearance 阶段确实会影响 alpha/depth 聚合行为”的前提。

### 总结感悟
- 这条 change 在工程接线层面并不难,甚至比 proposal 里预估的更轻,因为 V1 不需要先打开 dataloader depth 契约。
- 真正需要警惕的不是“能不能接进去”,而是不要把 baseline_render self-anchor 说成几何纠偏方案。
- 更准确的定位应是: 它可以抑制 RGB-only / SR pressure 带来的进一步深度漂移,但不会天然修复 baseline 初始化本身的厚表面。

## [2026-03-09 04:40:00 UTC] 任务名称: 追踪 `moge_version` 默认行为的实际调用链

### 任务内容
- 沿 README 常用命令和实际入口脚本追踪 `moge_version` 默认值的传播路径
- 区分 CLI 默认值、内建模型默认值、以及本地 checkpoint 自动识别三层语义
- 确认 `sample.py` 是否继续参与 `MoGe` 版本选择

### 完成过程
- 回读了六文件上下文,确认此前已有 `MoGe v1/v2` 相关结论,避免重复误判。
- 核对了 `README.md`、`scripts/bash/static_sdg.sh`、`gen3c_single_image.py`、`gen3c_single_image_sdg.py`、`gen3c_persistent.py`、`inference_utils.py`。
- 用结构化搜索和代码阅读确认:
  - 这些入口统一复用 `add_moge_arguments(...)`
  - `--moge_version` 的 parser 默认值是 `auto`
  - `auto` 在未显式指定模型时会解析到 `Ruicheng/moge-2-vitl`
  - `sample.py` 本身没有 `moge` 相关逻辑
- 额外确认 `gen3c_persistent.py` 复用 `gen3c_single_image.create_parser()`,因此也继承相同默认行为。

### 总结感悟
- 这类“默认值”最好拆成三层看: CLI 表面默认、内部解析默认、以及本地权重自动识别。
- 这次如果只看 `argparse default`,会得到 `auto`; 但如果看 README 实际命令最终效果,又确实是 `v2`。两种说法都各自只对了一半。
- 后续再讨论 `MoGe v1/v2` 差异时,应优先先问清楚用户跑的是哪一个入口,以及有没有显式传本地 checkpoint。

## [2026-03-09 04:45:00] 任务名称: 默认启用 auto_center_depth

### 任务内容
- 将单图 / static SDG 入口的 `auto_center_depth` 改为默认开启
- 保留显式关闭开关,避免完全失去旧的固定 `center_depth=1.0` 行为
- 补充测试并同步 README 文案

### 完成过程
- 在 `cosmos_predict1/diffusion/inference/inference_utils.py` 中:
  - 将 `auto_center_depth=True` 写入 parser 默认值
  - 新增 `--no_auto_center_depth` 作为显式关闭入口
  - 更新帮助文案,说明自动中心深度现在默认开启
- 在 `tests/test_camera_trajectory_center_depth.py` 中新增回归测试:
  - 默认解析结果应为 `auto_center_depth=True`
  - 显式传 `--no_auto_center_depth` 应回退为 `False`
- 在 `README.md` 中同步说明:
  - 单图/static SDG 路径默认启用 `auto_center_depth`
  - 需要旧行为时应显式传 `--no_auto_center_depth`

### 总结感悟
- 这类“改默认值”最容易埋下的雷,不是代码改动本身,而是把旧行为彻底变成无法恢复。
- 这次通过“默认开启 + 显式关闭开关 + 测试锁死”的方式,把新默认和历史兼容都保住了。
- 运行 Python 测试时要注意环境差异: 当前相机轨迹测试依赖 `warp`,并且在 `pixi` 环境下仍需显式补 `PYTHONPATH`。

## 2026-03-09 任务名称: 排查 static view 索引与 dataset_registry 对应关系

### 任务内容
- 核对 `static_view_indices_fixed=['5','0','1','2','3','4']` 与 `dataset_registry` 中 `sampling_buckets`、`start_view_idx` 是否存在错位.
- 覆盖 `src/models/data/provider.py`、`src/models/data/radym.py`、`src/models/data/registry.py` 与相关测试.

### 完成过程
- 回读历史实验记录,确认仓库曾经主动把静态轨迹顺序改成 `5,0,1,2,3,4` 作为推理输入顺序.
- 静态阅读确认:
  - fixed 模式直接使用 `static_view_indices_fixed`
  - `sampling_buckets` 只在 `random_bucket` 模式下使用
  - `start_view_idx` 只影响随机 bucket 偏移与默认起始 view
- 动态验证:
  - 使用 `.pixi/envs/default/bin/python` 实例化 `Provider('lyra_static_demo_generated', training=False)`
  - 实测 `provider._get_indices_static(0)` 返回 `input_view_indices = ['5', '0', '1', '2', '3', '4']`
  - 资产目录 `assets/demo/static/diffusion_output_generated/{0..5}/rgb/00172.mp4` 全部存在

### 总结感悟
- 当前不是“fixed 列表位置要和 bucket 位置对齐”的设计, 而是 fixed 列表本身就是最终 view ID 顺序.
- 以后只有在切回 `random_bucket` 或修改 view 编号起点时, 才需要重新审查 `sampling_buckets` 和 `start_view_idx`.
