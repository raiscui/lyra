# 笔记: 2026-03-10 磁盘垃圾盘点

## 六文件摘要(用于决定如何沉淀知识)

- 任务目标(`task_plan` 旧批次): 旧主线集中在 `refinement_v2`、`selective SR`、`Stage 2B`、`Phase A/B` 等模型实验与实现验证。
- 关键决定(旧 `task_plan` / `WORKLOG`): 旧批次已经形成稳定的“测试先行 + 真实实验复核 + 六文件留痕”节奏, 并多次把错误方向通过反证尽快关掉。
- 关键发现(旧 `notes` / `EPIPHANY_LOG`): 旧主线里多次出现“不要把候选假设说成根因”“要区分 durable state 与 runtime cache”“指标持平不等于实现无效”这几条稳定规律。
- 实际变更(旧 `WORKLOG` / `ERRORFIX`): 最近一批工作主要围绕 `refinement_v2` 的 warm-start、patch supervision、fidelity diagnostics、`Phase A` 路由重排与相关测试。
- 暂缓事项(旧 `LATER_PLANS`): 主要是 selective SR 后续参数校准、`Phase B`、以及若干实验 runbook 与诊断增强。
- 重大风险(旧 `EPIPHANY_LOG`): 旧主线强调 resume 语义、设备一致性、dense render meta 显存墙、以及相机默认值变更的回归风险。
- 可复用点候选:
  - 新任务若与旧主线弱相关, 应及时整体续档六文件, 避免污染判断。
  - 历史文件在归档前, 先做最小持续学习摘要, 再移动进 `archive/`。
  - 对排查类任务, 继续沿用“现象 -> 假设 -> 验证证据 -> 结论”的表达方式。
  - 当 Markdown 正文含反引号时, 不要把整段脚本再包进外层单引号 `bash -lc ...`, 更稳的写法是用 `python3` 写文件或严格使用单引号 heredoc。
- 最适合写到哪里: 本轮先沉淀在新的 `notes.md`, 暂不需要额外更新仓库 `docs/` 或 `AGENTS.md`。
- 需要同步的现有 `docs/` / `specs/` / plan 文档: 无。
- 是否需要新增或更新 `docs/` / `specs/` / plan 文档: 否。
- 是否提取/更新 skill: 否, 当前没有形成新的跨任务技巧。

## [2026-03-10 00:00:00 UTC] 主题: Stage 3SR patch supervision 静态链路梳理

### 现象
- `SceneBundle` 同时携带 native 监督(`gt_images`, `intrinsics`)与 reference 监督(`reference_images`, `intrinsics_ref`)字段, 说明 SR supervision 的上游数据在进入 runner 前就已准备好.
- `run_stage2a()` 在 enhanced 模式下固定顺序是: `run_stage3a_native_cleanup()` -> `run_phase3s_build_sr_selection()` -> `run_stage3sr_selective_patch()`.
- `run_phase3s_build_sr_selection()` 会在 native render 的基础上, 利用 `render_meta` 和 `WeightBuilder.build_sr_selection_weight()` 构造 reference 分辨率的 `sr_selection_map`.
- `Stage 3SR` 真正的监督计算统一收敛到 `_compute_patch_losses()`, 单卡、多卡、warm-start、Stage 2B 都复用这一个入口.
- 单热点约束来自 `sample_patch_windows()`: 每个 `[B, V]` 只取一个 argmax hotspot, 输出 shape 固定为 `[B, V, 4]`.

### 关键证据
- 上游 reference 构造: `src/refinement_v2/data_loader.py:738`, `src/refinement_v2/data_loader.py:775`, `src/refinement_v2/data_loader.py:776`
- Stage 2A -> Phase 3S -> Stage 3SR 顺序: `src/refinement_v2/runner.py:1722`, `src/refinement_v2/runner.py:1745`
- Phase 3S 生成 reference 分辨率 selection map: `src/refinement_v2/runner.py:1483`, `src/refinement_v2/runner.py:1506`, `src/refinement_v2/runner.py:1511`, `src/refinement_v2/runner.py:1518`
- selection map 投影逻辑: `src/refinement_v2/weight_builder.py:258`
- patch 主链路: `src/refinement_v2/runner.py:807`, `src/refinement_v2/runner.py:829`, `src/refinement_v2/runner.py:842`, `src/refinement_v2/runner.py:884`, `src/refinement_v2/runner.py:898`, `src/refinement_v2/runner.py:953`
- 权重组合与损失: `src/refinement_v2/weight_builder.py:383`, `src/refinement_v2/losses.py:54`, `src/refinement_v2/losses.py:214`
- 单卡 Stage 3SR 接线: `src/refinement_v2/runner.py:1373`
- 多卡 Stage 3SR 接线: `src/refinement_v2/runner.py:1157`, `src/refinement_v2/runner.py:1208`

### 当前判断
- 已有“reference 分辨率局部 patch 渲染”入口: `render_patch_prediction()`.
- 已有“任意分辨率 scene 渲染”底层能力: `render_scene()` + `_get_renderer_for_scene()`, 但没有现成的“整图 reference 分辨率渲染 helper”.
- 还没有现成“多 patch 渲染”入口, 因为 patch window / gather / intrinsics / render helper 全都默认单 patch 形状 `[B, V, 4]`.

### 最可能的最小挂点(静态推断)
- 第一挂点: `src/refinement_v2/runner.py:953` 的 `_compute_patch_losses()`. 这里已经拿到了 residual、native robust weight、reference 级 `sr_selection_map`, 且所有阶段都汇聚到这里.
- 第二挂点: `src/refinement_v2/runner.py:807` 的 `sample_patch_windows()`. 这里是“单热点”策略的根.
- 如果目标是更大 coverage 而不是严格多 patch 列表, 可以优先保留 Phase 3S 和 selection map 生成逻辑不动, 只把 `_compute_patch_losses()` 的渲染目标从“单 patch crop”扩成“多 patch / 更大区域 / 整图 reference render”.

### 2026-03-10 03:30:28 UTC

- 第一轮现象: `df` 显示根盘约 `242G` 已用, 但容器内 `du` 可见目录总量仅约 `293M`。
- 当前主假设: 大头不在当前可见工作区, 更可能位于容器底层镜像层或宿主机的 overlay 数据。
- 下一步验证: 继续盘点容器内可清理缓存, 并补查 overlay 挂载信息。

### 2026-03-10 03:31:09 UTC

- 第二轮现象: 小范围 `du -x` 被挂载边界遮住了真实大头。
- 已观察到的大目录候选:
  - `~/.cache/huggingface` 约 `90G`
  - `~/.cache/pip` 约 `7.4G`
  - `/tmp` 约 `5.2G`
  - `/workspace/lyra/.pixi` 约 `14G`
  - `/workspace/lyra/outputs` 约 `3.5G`
  - 其他仓库的 `.pixi` 也分别约 `8.3G`、`12G`
- 下一步验证: 核对这些路径是否为独立挂载、缓存目录还是项目产物目录, 再做可清理性分类。

## [2026-03-10 00:28:00 UTC] 更正: 之前“当前没有多 patch 入口”的判断不成立

### 被推翻的上一判断
- 我上一条静态笔记里写了“当前没有现成多 patch 渲染入口”.

### 推翻证据
- `src/refinement_v2/runner.py:857` 已新增 `sample_sr_patch_window_sets(...)`, 明确按 `sr_patches_per_view` 选多个 patch.
- `src/refinement_v2/runner.py:1060` 的 `_compute_patch_losses(...)` 已经循环 `for patch_windows_native in patch_window_sets`, 并在 `src/refinement_v2/runner.py:1126` 对多 patch loss 做平均.
- `src/refinement_v2/config.py:77` / `src/refinement_v2/config.py:313` 已经暴露 `sr_patches_per_view` 参数面.
- `tests/refinement_v2/test_patch_supervision.py:229` 与 `tests/refinement_v2/test_patch_supervision.py:294` 已经覆盖“按 selection priority 取多个 patch”与“真的渲染多个 patch set”.

### 更新后的结论
- 当前代码已经有“多 patch per view 的 Phase B 最小版”入口.
- 但它仍然不是“reference 分辨率整图渲染”入口.
- 如果还要继续扩大 coverage, 最可能的新增空间已经从“单 patch -> 多 patch”转移到:
  - patch priority 构造策略
  - patch set 数量/抑制策略
  - 或直接补一个 full-reference render 分支

### 2026-03-10 03:34:14 UTC

## 候选垃圾分类结论

### A. 相对安全, 优先考虑的清理候选

- `~/.cache/huggingface/hub` 约 `90G`
  - 其中 `models--google-t5--t5-11b` 单项约 `85G`
  - 性质: 下载缓存
  - 风险: 删除后需要重新下载, 但通常不影响项目结构
- `~/.cache/pip` 约 `7.4G`
  - `http-v2` 约 `7.0G`, `wheels` 约 `372M`
  - 性质: pip 下载与 wheel 缓存
  - 风险: 低
- `/tmp` 下多项历史临时目录
  - `longsplat-no-source` 约 `3.7G`
  - `tttLRM` 约 `1.1G`
  - `gradio` 约 `372M`
  - `LongSplat` 约 `114M`
  - `mermaid-validate` 约 `20M`
  - 性质: 临时 clone / 临时运行产物 / 调试目录
  - 风险: 低到中, 但清理前最好确认没有正在运行的进程依赖这些目录

### B. 可重建, 但删除成本较高的候选

- `/workspace/lyra/.pixi` 约 `14G`
- `/workspace/vipe/.pixi` 约 `12G`
- `/workspace/FastGS/.pixi` 约 `11G`
  - 性质: 各仓库独立 Pixi 环境, 当前都只有 `default` 环境
  - 风险: 删除后可重建, 但会花时间重新解环境与下载依赖
- `/workspace/FlashVSR-Pro/models/FlashVSR-v1.1` 约 `13G`
  - 其中 `.git` 约 `6.5G`, 模型权重本体约数 GB
  - 性质: 模型仓库 + 权重
  - 风险: 若当前还会使用 FlashVSR, 不建议直接删

### C. 很大, 但不应默认视为垃圾

- `/workspace/lyra/checkpoints` 约 `31G`
  - 其中 `Gen3C-Cosmos-7B/model.pt` 单文件约 `27G`
  - 性质: 主项目模型权重
  - 风险: 高, 删除会直接影响项目运行
- `/workspace/lyra/assets/demo` 约 `7.9G`
  - `dynamic` 约 `4.4G`, `static` 约 `3.5G`
  - 性质: 项目 demo 数据资产
  - 风险: 高

### D. 介于两者之间, 需要人工判断是否过期

- `/workspace/lyra/outputs` 约 `3.5G`
  - `flashvsr_reference` 约 `1012M`
  - `refine_v2` 约 `2.3G`
  - `refine_v2` 下多为 2026-03-09 与 2026-03-10 的实验输出目录
  - 性质: 实验结果与导出视频
  - 风险: 中, 若这些 run 证据已经不再需要, 可以清; 若还要做对比, 先别动

## 证据补充

- `lsof +L1` 没有发现 `>10M` 的已删除但仍被进程占用的文件。
- 顶层 `du -x /` 在这个 overlay 环境里会严重低估真实热点, 必须改成对具体大目录逐个排查。
- 对多个目录分别跑 `du` 时, 共享硬链接可能导致简单求和偏大, 因此报告中的容量只适合逐项判断, 不适合直接相加成总和。

## 2026-03-10 03:36:30 UTC `Phase B` first try: `sr-patches-per-view=4` OOM

### 现象

- 运行目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseB_multi4_sub8_iter20_20260310`
- 已完成产物:
  - `metrics_stage2a.json`
  - `metrics_phase3s.json`
  - `gaussian_fidelity_histogram.json`
  - `sr_selection_stats.json`
- 未完成产物:
  - `metrics_stage3sr.json`
  - `diagnostics.json`
- 真实错误栈定位到:
  - `render_patch_prediction()`
  - `src/rendering/gs.py`
  - `gsplat.rasterization()`
  - `torch.OutOfMemoryError`

### 当前结论

- 目前动态证据只支持一个保守结论:
  - `sr_patches_per_view=4` 对当前 full-view `sub8` + `patch-size=256` 太重
- 还不能把这解释成代码 bug
- 下一步最小验证应当是:
  - 只降到 `sr_patches_per_view=2`
  - 其它参数不变

## 2026-03-10 03:48:00 UTC `Phase B` first validation results

### 代码与测试

- 新增参数:
  - `sr_patches_per_view`
- 新增行为:
  - `sample_sr_patch_window_sets()`
  - 基于 reference priority 选择多个 patch set
  - `_compute_patch_losses()` 对多个 patch set 求平均
- 回归:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2/test_config.py tests/refinement_v2/test_patch_supervision.py tests/refinement_v2/test_weight_builder.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_phase0.py tests/refinement_v2/test_runner_stage2b.py`
  - 结果: `47 passed`

### 真实实验

#### `multi4`

- 目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseB_multi4_sub8_iter20_20260310`
- 现象:
  - `stage2a` 与 `phase3s` 正常完成
  - 一进入 `stage3sr` 的 reference patch render 即 OOM
- 当前结论:
  - `sr_patches_per_view=4` 对当前 full-view `sub8` + `patch-size=256` 太重

#### `multi2`

- 目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseB_multi2_sub8_iter20_20260310`
- 相对旧 baseline:
  - `PSNR: 23.996732 -> 23.997848` (`+0.001116`)
  - `residual_mean: 0.03452381 -> 0.03443952` (`-0.0000843`)
  - `sharpness: 0.00295948 -> 0.00295705` (`-0.00000244`)
- 相对 `Phase A` rerun:
  - `PSNR: 24.000970 -> 23.997848` (`-0.003122`)
  - `residual_mean: 0.03442931 -> 0.03443952` (`+0.0000102`)
  - `sharpness: 0.00295518 -> 0.00295705` (`+0.00000186`)
- 额外观察:
  - `sr_patch_sets_used = 2`
  - `phase3s` 的 fidelity / selection 分布与 `Phase A` 基本一致

### 结论

- `Phase B` 最小版已经把 coverage 从单 patch 扩到 multi-patch
- 但当前 multi-patch 还没有把最终收益显著拉高
- 因此问题更像已经从“有没有 coverage”进入了“coverage 该怎么配强度与目标”的阶段

## [2026-03-10 06:10:00 UTC] fidelity CLI 接线 + full-view `sub8` calibration

### 现象

- `WeightBuilder` 内部早就已经有 fidelity 相关超参数:
  - `fidelity_ratio_threshold`
  - `fidelity_sigmoid_k`
  - `fidelity_min_views`
  - `fidelity_opacity_threshold`
- 但在这轮之前, `config.py` / CLI / `RefinementRunner` 都没有把这些参数真正打通。
- 因此之前所有 `Phase A` / `Phase B` full-view `sub8` 真实结果, 实际都仍然在吃 `WeightBuilder` 默认值。

### 静态证据

- `src/refinement_v2/weight_builder.py` 已定义并消费上述 4 个参数。
- `src/refinement_v2/config.py` 本轮前没有对应 `StageHyperParams` 字段, 也没有对应 CLI 参数。
- `src/refinement_v2/runner.py` 本轮前初始化 `WeightBuilder(...)` 时也没有传这些 fidelity 参数。

### 本轮修复

- 在 `src/refinement_v2/config.py` 的 `StageHyperParams` 中新增:
  - `fidelity_ratio_threshold`
  - `fidelity_sigmoid_k`
  - `fidelity_min_views`
  - `fidelity_opacity_threshold`
- 在 CLI 中新增:
  - `--fidelity-ratio-threshold`
  - `--fidelity-sigmoid-k`
  - `--fidelity-min-views`
  - `--fidelity-opacity-threshold`
- 在 `src/refinement_v2/runner.py` 初始化 `WeightBuilder(...)` 时透传上述 4 个参数。
- 补测试:
  - `tests/refinement_v2/test_config.py`
  - `tests/refinement_v2/test_runner_stage2a.py`

### 回归验证

- 命令:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2/test_config.py tests/refinement_v2/test_runner_stage2a.py tests/refinement_v2/test_patch_supervision.py tests/refinement_v2/test_weight_builder.py tests/refinement_v2/test_runner_phase0.py tests/refinement_v2/test_runner_stage2b.py tests/refinement_v2/test_depth_anchor.py`
- 结果:
  - `58 passed`

### 最小 calibration 实验

- 命令口径:
  - full-view
  - external SR reference
  - `--target-subsample 8`
  - `--iters-stage2a 20`
  - 只改一项:
    - `--fidelity-ratio-threshold 1.1`
- 输出目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_thr1p1_sub8_iter20_20260310`

### 动态证据

- baseline:
  - `outputs/refine_v2/full_view_sr_stage3sr_long_sub8_iter20_20260309`
  - `PSNR = 23.996732275883772`
  - `residual_mean = 0.03452381119132042`
  - `sharpness = 0.002959481906145811`
  - `selection_mean = 0.1455942988395691`
- `Phase A` rerun:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_sub8_iter20_20260310`
  - `PSNR = 24.00097032415988`
  - `residual_mean = 0.03442930802702904`
  - `sharpness = 0.0029551826883107424`
  - `selection_mean = 0.015738628804683685`
  - `fidelity_mean = 0.3613118827342987`
- calibration(`threshold=1.1`):
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_thr1p1_sub8_iter20_20260310`
  - `PSNR = 24.000384521290037`
  - `residual_mean = 0.03444404900074005`
  - `sharpness = 0.002955049742013216`
  - `selection_mean = 0.04699167236685753`
  - `fidelity_mean = 0.10911799967288971`

### 对比结论

- 相对旧 baseline:
  - `PSNR +0.003652`
  - `residual_mean -0.0000798`
  - `sharpness -0.00000443`
- 相对 `Phase A` rerun:
  - `PSNR -0.000586`
  - `residual_mean +0.0000147`
  - `sharpness -0.000000133`
  - `selection_mean +0.031253`
  - `fidelity_mean -0.252194`
- 当前能确认的是:
  - 把 `fidelity_ratio_threshold` 从 `1.5` 降到 `1.1`, 确实把 selection coverage 明显放大了
  - 但最终指标仍然没有超过当前 `Phase A`
- 因此当前更像是:
  - routing 强度已经不是唯一瓶颈
  - 下一步若继续 Closest-to-SplatSuRe 主线, 应优先转向 `Phase C: HR render + LR consistency`

## [2026-03-10 06:20:00 UTC] 当前 `Stage 3SR` 是否仍在同时使用 native GT 与 6 路 SR reference

### 现象

- 用户提出了一个关键判断:
  - 既然 6 路超分视频和原始 6 路 GT 共享同一套相机参数, 是否不需要继续同时保留老的 native GT
  - 是否可以直接把 SR 6 视频当成新的 GT

### 静态证据

- loader 侧:
  - `src/refinement_v2/data_loader.py:752`
    - 如果给了 `reference_path`, 会单独构造 external reference supervision
  - `src/refinement_v2/data_loader.py:765`
    - `reference_mode=native` 时, `reference_images` 才会直接等于 `gt_images`
  - 这说明 current scene bundle 语义一直是:
    - `gt_images` = native LR 监督
    - `reference_images` = reference / SR 监督
- runner 侧:
  - `src/refinement_v2/runner.py:753`
    - `_get_reference_images()` 优先取 `reference_images`
  - `src/refinement_v2/runner.py:761`
    - `_get_gt_images()` 永远取 `gt_images`
  - `src/refinement_v2/runner.py:1479`
    - `Stage 2A / Stage 3SR` 主 RGB loss 仍然对 `gt_rgb = gt_images`
  - `src/refinement_v2/runner.py:1486`
    - `loss_rgb = compute_weighted_rgb_loss(pred_rgb, gt_rgb, ...)`
  - `src/refinement_v2/runner.py:1017`
    - patch 路径会从 `reference_images` gather `reference_patch`
  - `src/refinement_v2/runner.py:1127`
    - patch RGB loss 明确是 `pred_patch` 对 `reference_patch`

### 已验证结论

- 当前实现里, 是的, 还在同时使用两套监督:
  - native LR `gt_images`
  - 6 路 SR `reference_images`
- 当前 OOM 的直接热点仍然是:
  - reference patch render 路径
  - 尤其是 `sr_patches_per_view` 增大后
- 不是因为还保留了 native GT 这件事本身就触发了 OOM

### 对 `Phase C` 的启发

- 如果直接把 SR 6 视频当成新的唯一 GT, 那就等价于:
  - 放弃当前 native LR 主损失
  - 改成纯 HR 监督主线
- 这当然可以做, 但那已经不是“HR render + LR consistency”, 而是“HR-only supervision”。
- 更稳、更接近当前 `Phase C` 目标的做法应是:
  - SR 6 视频成为主监督
  - native LR 不再是并行主目标, 而是下采样一致性约束
- 也就是:
  - HR render 对 SR 6 视频做主损失
  - HR render downsample 后, 再和 native LR 做 consistency

## [2026-03-10 06:35:00 UTC] `Phase C` 方案与任务已按新共识改写

### 变更原因

- 用户已明确认可新的 `Phase C` 口径:
  - SR 6 视频升格为主监督
  - native LR 不再和 SR 并列做主损失
  - 而是退到 downsample consistency
- 这会直接改变 `Closest-to-SplatSuRe Track` 的方案定义和任务顺序。

### 已更新文档

- `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`

### 方案层更新点

- `Phase C` 现在明确排除了 `HR-only supervision` 版本。
- 新定义改成:
  - HR render 对 SR 6 视频做主监督
  - HR render 下采样后对 native LR 做 consistency
- 文档里额外写清了为什么不能直接彻底删除 native LR:
  - 否则更容易把 SR hallucination 直接写进 3D

### task 层更新点

- 文档中新增当前认可的 `Phase C` 实施任务:
  1. 补 full-frame HR render 路径
  2. 补 HR -> LR consistency 路径
  3. 重组 `Stage 3SR` 目标
  4. 补 diagnostics / metrics / tests
- `Recommended Execution Order` 已改成:
  1. 先做 `Phase C`
  2. 再做 `Phase D`
  3. 如有必要再回头重做 `Phase B`
  4. 最后评估 `Phase E`

### 当前结论

- 现在 backlog 已经从“继续调 A/B”正式切到“实现新定义的 Phase C”。
- 后续如果继续动代码, 就应该直接按文档里这 4 个任务往下做。


## [2026-03-10 06:30:00 UTC] Phase C 真实 OOM 诊断与 stream-shard 修正

### 现象1

- 命令口径:
  - full-view external SR
  - `--target-subsample 8`
  - `--lambda-hr-rgb 0.5`
  - `--lambda-lr-consistency 1.0`
  - `--reference-render-shard-views 1`
- 第一轮真实 smoke 目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_20260310`
- 真实报错:
  - `torch.OutOfMemoryError`
  - 栈落在 `gsplat.rasterization()`

### 假设1

- 当前 full-frame HR 路径虽然 reference render 是新分支, 但 native render 仍保留了 autograd 图。
- 因此 native 图和 HR 图同时驻留, 先在 renderer 阶段撞上 80G 显存峰值。

### 最小验证1

- 把 `_run_stage3sr_full_frame_hr()` 里的 native render 改成:
  - depth anchor 关闭时走 `torch.no_grad()`
- 其它口径不变, 重新跑同一条真实 `sub8` smoke。

### 现象2

- 第二轮目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_memfix_20260310`
- OOM 不再出现在 renderer.
- 新报错落在:
  - `_build_reference_supervision_weight_map()`
  - `WeightBuilder.combine_sr_weights()`
- 这说明“关掉 native render autograd”确实降低了第一层峰值, 但没有彻底解决问题。

### 被推翻的上一充分结论

- “只要让 native render 不参与反传, Phase C 就能在真实 `sub8` 跑通”不成立。

### 更新后的主假设

- 当前真正的问题是:
  - `_render_scene_serial_view_shards()` 虽然串行渲染了 HR shard
  - 但后面又把所有 shard `torch.cat(...)` 回了整块 HR tensor
  - 随后 full-frame 权重图和 LR consistency 也按全量 tensor 一次性构造
- 所以这其实只是“串行前向”, 不是“串行 loss/backward”, 峰值还是会回来。

### 最小验证2

- 新增 `_iter_scene_single_device_view_shards()`
- 把 `Phase C` full-frame HR 路径改成:
  1. native residual / weight 先算好
  2. HR render 按 shard 逐块前向
  3. HR loss / LR consistency 按 shard 逐块 backward
  4. 指标只在 shard 结束后 detach 到 CPU 再拼
- 再次重跑同口径真实 `sub8` smoke。

### 最终动态结论

- 第三轮目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_streamshard_20260310`
- 已完整生成:
  - `metrics_stage3sr.json`
  - `diagnostics.json`
  - `gaussians_stage3sr.ply`
  - `final_render.mp4`
- 关键结果:
  - `phase_reached = stage3sr`
  - `stage3sr_supervision_mode = full_frame_hr`
  - `psnr: 17.8840 -> 18.8914`
  - `residual_mean: 0.085080 -> 0.070438`
- 结论:
  - `Phase C` 真实 blocker 不是“full-frame HR render 根本跑不起来”
  - 而是必须把实现从“serial render + full concat”进一步推进到“serial render + serial loss/backward”


## [2026-03-10 06:45:00 UTC] Phase C lambda sweep 设计依据

### 已观察到的事实

- 当前最小真实 smoke:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_streamshard_20260310`
- 其中关键 loss 为:
  - `loss_hr_rgb = 0.0009198441`
  - `loss_lr_consistency = 0.0334337486`
- 因此:
  - `loss_hr_rgb / loss_lr_consistency ≈ 0.0275`

### 当前判断

- 如果继续固定 `lambda_lr_consistency = 1.0`, 那么 `lambda_hr_rgb` 仍停留在 `0.5 ~ 2.0` 这种区间的话, `Phase C` 实际上还是会被 LR consistency 主导。
- 为了让 sweep 真正有信息量, 本轮应直接尝试更高的 `lambda_hr_rgb`:
  - 至少进入 `8 / 16 / 32` 这种量级

### 本轮 sweep 策略

- 先固定:
  - `lambda_lr_consistency = 1.0`
- 再试三组:
  1. `lambda_hr_rgb = 8`
  2. `lambda_hr_rgb = 16`
  3. `lambda_hr_rgb = 32`
- 先全部跑 `iters-stage2a = 1` 的真实 `sub8` smoke
- 再从中挑一组进入更长 iter 对照


## [2026-03-10 07:05:00 UTC] Phase C 小 sweep 与 iter8 长跑结果

### 现象

- 小 sweep 跑了三组:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_sweep_hr8_lr1_sub8_20260310`
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_sweep_hr16_lr1_sub8_20260310`
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_sweep_hr32_lr1_sub8_20260310`
- 三组在 `1 iter` 下的关键指标几乎重合:
  - `psnr ≈ 18.8914`
  - `residual_mean ≈ 0.0704375`
  - `psnr_hr ≈ 17.84136`
- 随后长 iter 跑了:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr1_sub8_iter8_20260310`
- 该 run 最终结果:
  - `psnr = 22.2530`
  - `residual_mean = 0.044018`
  - `sharpness = 0.0020366`
  - `psnr_hr = 20.2697`
  - `residual_mean_hr = 0.059271`

### 当前判断

- `1 iter` smoke 的信息量已经明显不够, 因为它对 `lambda_hr_rgb=8/16/32` 基本无分辨力。
- 但一旦把 iter 拉长到 8, `Phase C` 的确开始出现明显收益:
  - 相对同轮最小 smoke:
    - `PSNR: 18.8914 -> 22.2530` (`+3.3616`)
    - `residual_mean: 0.070438 -> 0.044018` (`-0.026420`)
    - `psnr_hr: 17.8414 -> 20.2697` (`+2.4283`)
- 同时也要保守看待:
  - 它目前仍低于 `Phase A iter20`
  - 但这不是公平 apples-to-apples, 因为当前只跑到了 `iter8`

### 额外观察

- 在 `iter8` 最终点, 即便 `lambda_hr_rgb=32`, 仍然有:
  - `loss_hr_rgb = 0.000635`
  - `loss_lr_consistency = 0.020297`
- 这说明当前训练仍然主要受 LR consistency 主导。
- 换句话说:
  - 这轮已经证明 `Phase C` 可跑且能涨
  - 但还没证明当前 loss 配比已经真的让 HR supervision 成为主导

### 下一步入口

- 后续如果继续优先做参数, 下一条最有信息量的线更像是:
  - 固定 `lambda_hr_rgb=32`
  - 开始扫 `lambda_lr_consistency` 到 `0.5` 或 `0.25`
- 如果继续优先做阶段验证, 则应把当前配置拉到更长 iter 再与 `Phase A iter20` 比较

## [2026-03-10 07:47:33 UTC] Phase C iter20 长跑结果回收与对比

### 现象

- 之前仍在运行的长跑目录已经完整落盘:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter20_20260310`
- 已验证存在的核心产物:
  - `diagnostics.json`
  - `metrics_stage3sr.json`
  - `videos/final_render.mp4`
  - `gaussians/gaussians_stage3sr.ply`
- `metrics_stage3sr.json` 共 `20` 个 step, 最优点就是最终点。

### 关键结果

- `Phase C hr32 lr0.5 iter20`
  - `psnr = 23.546505890590247`
  - `residual_mean = 0.037821926176548004`
  - `sharpness = 0.002698792377486825`
  - `psnr_hr = 21.245776665329515`
  - `residual_mean_hr = 0.051848139613866806`
  - `psnr_native_render = 23.220529381240812`
  - `residual_mean_native_render = 0.03913893923163414`
- `Phase C hr32 lr1 iter8`
  - `psnr = 22.253002201596153`
  - `residual_mean = 0.04401839151978493`
  - `psnr_hr = 20.26965072517814`
  - `residual_mean_hr = 0.059270892292261124`
- `Phase A iter20`
  - `psnr = 24.00097032415988`
  - `residual_mean = 0.03442930802702904`
  - `sharpness = 0.0029551826883107424`

### 对比

- 相比 `Phase C hr32 lr1 iter8`:
  - `psnr +1.293504`
  - `residual_mean -0.006196`
  - `sharpness +0.000662`
  - `psnr_hr +0.976126`
  - `residual_mean_hr -0.007423`
  - `psnr_native_render +1.353694`
  - `residual_mean_native_render -0.006782`
- 相比 `Phase A iter20`:
  - `psnr -0.454464`
  - `residual_mean +0.003393`
  - `sharpness -0.000256`
- 因此当前最稳妥的表述是:
  - `Phase C` 还没有在 native 指标上超过 `Phase A iter20`
  - 但 gap 已经明显缩小
  - 同时 `Phase C` 额外提供了可观的 `HR-space` 指标

### 结论

- 当前主结论不是“`Phase C` 已经赢过 `Phase A`”。
- 当前主结论是:
  - `Phase C` 已从“工程可跑”推进到“长程收益已被动态证据验证”
  - 如果继续当前主线 backlog, 更值得切去 `Phase D`, 解决“最终导出仍是 native 输出”这个交付缺口

## [2026-03-10 08:01:24 UTC] Phase D 静态定位: 导出链路与最小切口

### 现象

- `final_render.mp4` 当前由 `RefinementRunner.export_final_outputs()` 统一导出:
  - `src/refinement_v2/runner.py:2610`
- `baseline_render.mp4` 与 `gt_reference.mp4` 当前由 `run_phase0()` -> `_export_baseline_visuals()` 导出:
  - `src/refinement_v2/runner.py:2326`
  - `src/refinement_v2/runner.py:2293`
- 真正落盘视频 / PNG 的底层函数在:
  - `src/refinement_v2/runner.py:2267` `_export_rgb_artifacts()`
  - `src/refinement_v2/diagnostics.py:189` `save_render_video()`
  - `src/refinement_v2/diagnostics.py:201` `save_render_snapshot()`

### 已确认的结构关系

- 当前最终 after 导出只走 native scene:
  - `export_final_outputs()` 内部直接调用 `_render_scene_for_evaluation(self.scene)`
  - 也就是 `src/refinement_v2/runner.py:2619`
- 当前代码里其实已经有可复用的 HR scene helper:
  - `_build_reference_render_scene()` 在 `src/refinement_v2/runner.py:969`
  - `_render_reference_scene_for_training()` 在 `src/refinement_v2/runner.py:1005`
- 这说明 `Phase D` 不需要从 0 重新发明 HR scene 概念.
- 更像是缺一条“reference-space evaluation/export”链路.

### 当前假设

- 最小正确切口不是重写 diagnostics writer.
- 更合理的切法是:
  1. 在 runner 里补一个 `reference-space evaluation render` helper
  2. 复用现有 `_export_rgb_artifacts()` 导出 `baseline_render_hr / gt_reference_hr / final_render_hr`
  3. 在最终 diagnostics summary 里显式新增 `baseline_hr / final_hr`
- 这样能保持:
  - 现有 native 导出不破坏
  - `Phase C` 已有的 `psnr_hr` 等字段继续可用
  - `Phase D` 再新增一层更稳定的最终交付指标

### 备选解释

- 另一条路是直接把 `_summarize_prediction()` 抽象成可加 suffix 的通用函数, 然后把所有 HR 指标都折叠进去.
- 但这会更广泛影响现有 stage metrics 语义, 回归面更大.
- 当前更像适合作为后续重构, 不是这轮最小落地切口.

### 当前结论

- 静态证据已足够支持先做 runner 层最小改动:
  - 新增 HR evaluation render helper
  - 新增 HR artifacts export
  - 新增 final summary 中的 `baseline_hr / final_hr`
- diagnostics writer 底层无需大改.

## [2026-03-10 08:11:59 UTC] Phase D 最小实现与验证

### 现象

- 用户在 `Phase C` 收口后继续推进, 当前 backlog 的下一步是 `Phase D`。
- 代码阅读确认:
  - 训练期已经有 `reference-space` scene builder 和 HR-space metrics
  - 但最终导出仍只落 native `final_render.mp4`
- 这说明当前缺口更偏向“导出层没跟上”, 而不是“训练层还没有 HR 信息”。

### 主假设

- `Phase D` 的最小正确实现, 不该再另起一套新 renderer 或新 scene 体系。
- 更合适的做法是:
  - 复用现有 `reference-space` scene builder
  - 在 baseline/final export 时额外走一条 evaluation/export 版的 HR render
  - 再把 native / hr 的摘要显式拆到 `diagnostics.json`

### 验证计划

- 静态验证:
  - 改 `src/refinement_v2/runner.py`
  - 改测试夹具 `tests/refinement_v2/helpers.py`
  - 补 `Phase 0` 与 `stop_after=stage2a` 的 HR 导出测试
- 动态验证:
  - 跑定向测试
  - 再跑 `tests/refinement_v2` 全套回归
  - 尝试真实 full-view `sub8` smoke

### 结论

- 静态证据成立:
  - 最小切口就是复用 `reference-space` 现有 helper, 不需要二次发明导出管线
- 动态证据成立:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2` 通过, 共 `113 passed`
- 真实 smoke 结论也明确:
  - 两次都在 `Phase 0` baseline render OOM
  - 当时 GPU 上外部进程约占 `49 GiB`
  - 因此这不能当成 `Phase D` 代码回归证据

## [2026-03-10 08:13:02 UTC] Phase D 实现后的动态验证结论

### 现象

- 定向回归已通过:
  - `29 passed`
- 全量回归已通过:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2`
  - `113 passed`
- 真实 full-view external SR `phase0` smoke 也已成功:
  - `outputs/refine_v2/phaseD_phase0_hr_export_smoke_20260310`

### 关键动态证据

- `diagnostics.json` 中当前真实包含:
  - `native_hw = [704, 1280]`
  - `reference_hw = [1408, 2560]`
  - `baseline_hr`
  - `final_hr`
  - `psnr_gain_hr`
  - `sharpness_gain_hr`
  - `residual_mean_hr_drop`
- 已确认真实落盘:
  - `videos/baseline_render_hr.mp4`
  - `videos/gt_reference_hr.mp4`
  - `videos/final_render_hr.mp4`
  - 对应的 `*_hr_frame_0000.png`

### 结论

- `Phase D` 现在已经从“代码层支持”推进到“真实资产上可见、可交付”。
- 这意味着后续再跑 `Phase C` 长训时, 不会再只有 LR after 视频作为最终观察窗口。
- 当前 backlog 应从“先补导出”切回“继续压优化质量”。

## [2026-03-10 08:13:38 UTC] Phase D 动态验证口径修正

### 被新证据推翻的上一表述

- “`Phase D` 当前还没有真实 smoke 级动态证据”这个表述不成立。

### 推翻它的证据

- 目录 `outputs/refine_v2/phaseD_phase0_hr_export_smoke_20260310` 已真实存在。
- 已读到的动态证据包括:
  - `phase_reached = phase0`
  - `native_hw = [704, 1280]`
  - `reference_hw = [1408, 2560]`
  - `artifacts` 中同时存在:
    - `baseline_render_hr_video`
    - `gt_reference_hr_video`
    - `final_render_hr_video`
  - `diagnostics.json` 中存在:
    - `baseline_hr`
    - `final_hr`
    - `psnr_gain_hr`
    - `sharpness_gain_hr`
    - `residual_mean_hr_drop`

### 修正后的结论

- `Phase D` 已经有一条 `phase0` 级真实动态证据。
- 本轮被外部显存占用挡住的是更重的:
  - full-view
  - `target-subsample 8`
  - `stop_after=stage2a`
  这一条 smoke。
- 所以现在最准确的表述应该是:
  - `Phase D` 已被单测 + 全套回归 + `phase0` 真实 smoke 支撑
  - 仍待补的是更重的 `stage2a` 级真实 smoke

## [2026-03-10 08:53:00 UTC] Phase C iter32 收口与 backlog 重排

### 现象

- 真实输出目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter32_20260310`
- 重新读取 `diagnostics.json` 与 `metrics_stage3sr.json` 后确认:
  - `phase_reached = stage3sr`
  - `stopped_reason = metrics_plateau`
  - `psnr = 24.15456836140987`
  - `residual_mean = 0.03491737321019173`
  - `sharpness = 0.003191302763298154`
  - `psnr_hr = 21.61596190185022`
  - `residual_mean_hr = 0.04965432360768318`
- 同目录里已确认存在:
  - `videos/final_render.mp4`
  - `videos/final_render_hr.mp4`
  - `videos/baseline_render_hr.mp4`
  - `diagnostics.json`
  - `metrics_stage3sr.json`

### 当前主假设

- `Phase C` 现在已经不再需要继续证明“它是否成立”。
- 真正剩下的问题, 已经变成:
  - 在保住当前 `HR-space` 收益的前提下
  - 能不能把 native `residual_mean` 也一起压过 `Phase A iter20`
- 因此下一轮最该验证的, 不是回到旧架构分支, 而是继续围绕 `lambda_lr_consistency≈0.5` 做近邻 sweep。

### 最强备选解释

- 也可能这轮领先主要只是来自更长 iter, 而不是 `lambda_lr_consistency=0.5` 本身已经处在最优附近。
- 如果后续 `0.4 / 0.6` 两个近邻点都不能进一步缩小 native `residual_mean` 差距, 那就说明当前瓶颈可能已经不在 loss 配比, 而在 `Phase E` 级别的结构自由度。

### 验证命令与关键输出

- 读取三组 run 的最终摘要:
  - `python3` 读取:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter20_20260310/diagnostics.json`
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter32_20260310/diagnostics.json`
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_sub8_iter20_20260310/diagnostics.json`
- 关键对比结论:
  - `Phase C iter32 - Phase C iter20`:
    - `psnr +0.608062`
    - `residual_mean -0.002905`
    - `sharpness +0.000493`
    - `psnr_hr +0.370185`
    - `residual_mean_hr -0.002194`
  - `Phase C iter32 - Phase A iter20`:
    - `psnr +0.153598`
    - `residual_mean +0.000488`
    - `sharpness +0.000236`
- 读取 `metrics_stage3sr.json` 最后一个点:
  - `loss_hr_rgb = 0.00040636223411638634`
  - `loss_lr_consistency = 0.01564438372345952`
  - `stage3sr_supervision_mode = full_frame_hr`

### 结论

- 已验证结论:
  - `Phase C hr32 lr0.5 sub8 iter32` 已经在 native `psnr` 上超过 `Phase A iter20`。
  - 同时它保留了更强的 `HR-space` 指标和最终 HR 导出产物。
  - native `residual_mean` 仍只差 `+0.000488`, 已经不是“方向不成立”, 而是“最后一点 trade-off 怎么压”。
- 因此 backlog 需要从:
  - “先证明 `Phase C` / 先补 `Phase D`”
  切换到:
  - “继续压 `Phase C` 的 residual frontier”
  - “必要时再评估 `Phase E`”

## 六文件摘要(用于决定如何沉淀知识)

- 任务目标(`task_plan.md`):
  - 回读当前状态, 收口 `Phase C iter32`, 并把 backlog 调整到新的证据基础上。
- 关键决定(`task_plan.md`):
  - 先续档超过 1000 行的 `task_plan.md`, 再同步文档和后续实验计划。
- 关键发现(`notes.md`):
  - `Phase C iter32` 已在 native `psnr` 上超过 `Phase A iter20`, 但 `residual_mean` 仍略高。
- 实际变更(`WORKLOG.md`):
  - 更新了 `docs/cmd.md` 与长期计划文档, 并归档了已回读的历史六文件。
- 暂缓事项 / 后续方向(`LATER_PLANS.md`):
  - 优先做 `lambda_lr_consistency≈0.5` 的近邻 sweep, 再决定是否进入 `Phase E`。
- 错误与根因(`ERRORFIX.md`):
  - 本轮没有新的 bugfix。
- 重大风险 / 重要规律(`EPIPHANY_LOG.md`):
  - 当主线已经跨过 baseline 时, backlog 不应再停留在“是否成立”的旧问题上。
- 可复用点候选:
  1. `Phase C` 一旦在 native `psnr` 上反超 baseline, 后续优先目标应改成 residual frontier, 不是回到旧 objective 争论。
  2. 超过 1000 行的六文件要及时续档, 否则后续决策会被旧上下文淹没。
- 最适合写到哪里:
  - `docs/cmd.md`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
  - `EPIPHANY_LOG.md`
- 需要同步的现有文档:
  - `docs/cmd.md`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
- 是否需要新增或更新 skill:
  - 否。本轮知识更偏项目当前 backlog 决策, 现阶段写回文档和 EPIPHANY 更合适。

## [2026-03-10 09:05:30 UTC] `lambda_lr_consistency=0.6` 近邻实验的中间证据

### 现象

- 用户在 `lr=0.6, iter32, sub8` 运行过程中主动中断了实验。
- 当前中间产物 `metrics_stage3sr.json` 已存在, 最新长度为 `26`。
- 最新中间点读数为:
  - `psnr = 24.109913076209125`
  - `residual_mean = 0.03496415540575981`
  - `sharpness = 0.0030633641872555017`
  - `psnr_hr = 21.528477663293394`
  - `residual_mean_hr = 0.050447121262550354`
  - `loss_lr_consistency = 0.015626876149326566`

### 初步结论

- 这条 `0.6` 近邻点到中途为止没有表现出“明显压过 `0.5 iter32`”的证据。
- 和已完成的 `0.5 iter32` 相比, 它当前是:
  - native `psnr` 略低
  - native `residual_mean` 略高
  - `HR-space` 指标也略低
- 因此现阶段还不能据此得出“继续加 LR consistency 就能把 residual 压过去”的结论。
- 但同样也不能从这条中途中断 run 直接反推出“应当彻底丢掉 LR consistency”。

## [2026-03-10 09:19:30 UTC] `HR-only` 对照完成: 直接移除 LR consistency 会显著伤害 native 指标

### 现象

- 真实输出目录:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p0_sub8_iter32_20260310`
- 最终 `diagnostics.json` 显示:
  - `phase_reached = stage3sr`
  - `stopped_reason = metrics_plateau`
  - `psnr = 22.271083371818634`
  - `residual_mean = 0.049804605543613434`
  - `sharpness = 0.00288753816857934`
  - `psnr_hr = 21.283824302586495`
  - `residual_mean_hr = 0.04892662912607193`

### 对比

- 相比 `lr=0.5 iter32`:
  - `psnr -1.883485`
  - `residual_mean +0.014887`
  - `sharpness -0.000304`
  - `psnr_hr -0.332138`
  - `residual_mean_hr -0.000728`
- 相比 `Phase A iter20`:
  - `psnr -1.729887`
  - `residual_mean +0.015375`
  - `sharpness -0.000068`

### 结论

- 已验证结论:
  - 当前主线里, 直接把 `lambda_lr_consistency` 设为 `0` 会显著破坏 native-space 指标。
  - 它并没有换来更强的 `HR-space` 胜利, 反而 `psnr_hr` 也更低。
- 这条证据说明:
  - `LR consistency` 不是无意义累赘。
  - 它现在仍然在提供关键的约束作用。
- 因此下一步更合理的是:
  - 继续补 `0.4` 近邻点
  - 而不是把主线直接切成纯 `HR-only`

## [2026-03-10 09:49:30 UTC] Phase E 最小实现与真实 smoke

### 现象

- 用户要求暂停 `Phase C` 参数调优, 先实现 `Phase E`。
- 本轮最小实现选择的是:
  - 不重写第二套 geometry pipeline
  - 直接复用 `Phase C` 的 full-frame HR reference-space 主监督路径
  - 在其后新增 `stage3b`, 允许有限 geometry 更新
- 代码层新增了:
  - `enable_stage3b`
  - `StageController.should_enter_stage3b(...)`
  - `run_stage3b()`
  - `compute_stage3b_losses(...)`

### 静态验证

- `python3 -m py_compile` 通过:
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/losses.py`
  - `src/refinement_v2/stage_controller.py`
  - `src/refinement_v2/runner.py`
  - `tests/refinement_v2/test_runner_stage3b.py`
- 定向测试:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2/test_config.py tests/refinement_v2/test_stage_controller.py tests/refinement_v2/test_runner_stage3b.py`
  - 结果: `25 passed`
- 全量回归:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2`
  - 结果: `117 passed`

### 动态验证

- 真实 smoke 目录:
  - `outputs/refine_v2/phaseE_stage3b_smoke_sub8_20260310`
- 命令口径:
  - `--enable-stage3b`
  - `--stage2a-mode enhanced`
  - `--lambda-hr-rgb 32`
  - `--lambda-lr-consistency 0.5`
  - `--target-subsample 8`
  - `--iters-stage2a 2`
  - `--iters-stage2b 2`
  - `--stop-after stage3b`
- 已确认:
  - `diagnostics.json` 存在
  - `phase_reached = stage3b`
  - `stopped_reason = metrics_plateau`
- 同一条 run 内部对比:
  - `stage3sr` 最后一点:
    - `psnr = 19.628720259216244`
    - `residual_mean = 0.06401330977678299`
    - `psnr_hr = 18.475841290772486`
    - `residual_mean_hr = 0.07501371204853058`
  - `stage3b` 最后一点:
    - `psnr = 20.43559102089945`
    - `residual_mean = 0.05679432675242424`
    - `psnr_hr = 19.01363852913593`
    - `residual_mean_hr = 0.06960125267505646`

### 结论

- 已验证结论:
  - `Phase E` 的最小版本已经落地, 不再只是文档里的概念。
  - 它能在真实 full-view `sub8` 资产上实际进入 `stage3b`。
  - 在最小 smoke 里, `stage3b` 相比同一条 run 的 `stage3sr` 已经带来正向改进:
    - `psnr +0.806871`
    - `residual_mean -0.007219`
    - `psnr_hr +0.537797`
    - `residual_mean_hr -0.005412`
- 当前限制也要明确:
  - 这还是最小版 `Phase E`
  - 仍复用了 `iters_stage2b`、`lambda_means_anchor`、`lambda_rotation_reg`、`means_delta_cap`
  - 还没有独立的 `stage3b` 专属超参数面

## [2026-03-10 10:20:00 UTC] Phase E 继续补 `stage3b` 独立超参数面

### 现象

- `Phase E` 的最小版 `stage3b` 已经能跑, 但运行期仍复用:
  - `iters_stage2b`
  - `lambda_means_anchor`
  - `lambda_rotation_reg`
  - `means_delta_cap`
- 这会让后续 `Phase E` 长跑难以做纯净归因。

### 假设

- 如果先把 `stage3b` 的 iteration / geometry regularizer / means clamp 从 `stage2b` 解绑,
- 后续 `Phase E` 的长跑与 calibration 会更容易解释, 也更不容易被旧 limited geometry 配置污染。

### 实现

- `src/refinement_v2/config.py`
  - 新增 `iters_stage3b`
  - 新增 `lambda_means_anchor_stage3b`
  - 新增 `lambda_rotation_reg_stage3b`
  - 新增 `means_delta_cap_stage3b`
  - 新增 CLI:
    - `--iters-stage3b`
    - `--lambda-means-anchor-stage3b`
    - `--lambda-rotation-reg-stage3b`
    - `--means-delta-cap`
    - `--means-delta-cap-stage3b`
  - 用 `__post_init__` 让 `stage3b` 默认继承旧共享 geometry 参数, 保持兼容
- `src/refinement_v2/runner.py`
  - 新增 `_resolve_geometry_stage_hparams(...)`
  - `stage3b` 改为读取独立 iteration / regularizer / clamp 配置
  - `stage2b` 继续走历史配置, 但也统一复用同一个解析 helper
  - metrics 里新增运行时证据:
    - `iters_budget`
    - `lambda_means_anchor_active`
    - `lambda_rotation_reg_active`
    - `means_delta_cap_active`
- `src/refinement_v2/gaussian_adapter.py`
  - `clamp_stage_constraints("stage3b", ...)` 现在优先读取 `means_delta_cap_stage3b`

### 静态验证

- `python3 -m py_compile src/refinement_v2/config.py src/refinement_v2/gaussian_adapter.py src/refinement_v2/runner.py tests/refinement_v2/test_config.py tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_runner_stage3b.py`
  - 通过
- `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2/test_config.py tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_runner_stage3b.py tests/refinement_v2/test_stage_controller.py`
  - 结果: `30 passed`
- `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2`
  - 结果: `119 passed`

### 动态验证

- 真实 smoke 目录:
  - `outputs/refine_v2/phaseE_stage3b_hparams_smoke_sub8_20260310`
- 启动前显存:
  - `0 MiB / 81920 MiB`
- 关键 CLI:
  - `--iters-stage2b 1`
  - `--iters-stage3b 2`
  - `--lambda-means-anchor 0.0`
  - `--lambda-means-anchor-stage3b 0.02`
  - `--lambda-rotation-reg 0.0`
  - `--lambda-rotation-reg-stage3b 0.02`
  - `--means-delta-cap 0.03`
  - `--means-delta-cap-stage3b 0.01`
- `diagnostics.json` 已确认:
  - `phase_reached = stage3b`
  - `stopped_reason = metrics_plateau`
- `metrics_stage3b.json` 最后一点已明确记录:
  - `iters_budget = 2`
  - `lambda_means_anchor_active = 0.02`
  - `lambda_rotation_reg_active = 0.02`
  - `means_delta_cap_active = 0.01`
- 同 run 内 `stage3b` 相比 `stage3sr`:
  - `psnr 19.628720 -> 20.435591`
  - `residual_mean 0.064013 -> 0.056794`
  - `psnr_hr 18.475841 -> 19.013638`
  - `residual_mean_hr 0.075014 -> 0.069601`

### 结论

- 已验证结论:
  - `stage3b` 的独立超参数面已经真实落地, 不只是 CLI 层的空壳。
  - 这次改动没有破坏 `tests/refinement_v2` 主线回归。
- 当前下一步最有信息量的不是再补配置, 而是跑一条更长的 `stage3b` apples-to-apples 对照。

## [2026-03-10 11:22:00 UTC] Phase E 长程与 continuation 收口

### 现象 1: 更长 auto-gate `Phase E` run 没有进入 `stage3b`

- 目录:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_hr32_lr0p5_sub8_iter32_20260310`
- 最终结果:
  - `phase_reached = stage3sr`
  - `stopped_reason = metrics_plateau`
  - `psnr = 24.154513879928512`
  - `residual_mean = 0.034917574375867844`
  - `psnr_hr = 21.62737934559564`
  - `residual_mean_hr = 0.04957122728228569`

### 假设 1

- 这不是 `stage3b` 跑挂了, 而是 auto gate 没放行。

### 静态证据

- `src/refinement_v2/runner.py:1803`
  - `local_overlap_persistent = residual_mean > 0.045`
- `src/refinement_v2/stage_controller.py:59`
  - `should_enter_stage3b(...)` 需要 `need_geometry=True` 且 `local_overlap_persistent=True`

### 动态证据

- source run 的 `state/latest.pt` 已确认:
  - `stage_name = stage3sr`
  - `stage3sr_completed = True`
  - `phase3s_completed = True`
  - `need_geometry = False`
  - `local_overlap_persistent = False`
- 这与最终 `residual_mean = 0.034917574375867844 < 0.045` 一致。

### 结论 1

- 已验证结论:
  - 更长 auto path 停在 `stage3sr`, 不能解释成“`stage3b` 跑了但没收益”。
  - 更准确的说法是: `stage3sr` 已经把 overlap 压到当前 geometry gate 以下。

### 现象 2: 从同一条 `stage3sr` 末态继续接 `stage3b`, 指标还能继续改善

- 手工 continuation:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
- 正式 CLI continuation:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
- 两条 continuation 都收敛到几乎同一结果:
  - 手工:
    - `psnr = 24.633636263443535`
    - `residual_mean = 0.03258064016699791`
    - `psnr_hr = 21.93410154162478`
    - `residual_mean_hr = 0.04774709418416023`
  - CLI:
    - `psnr = 24.633623626096842`
    - `residual_mean = 0.03258058801293373`
    - `psnr_hr = 21.934121746007385`
    - `residual_mean_hr = 0.047747090458869934`

### 对比

- continuation 相比 gate 停在 `stage3sr` 的结果:
  - `psnr +0.479122`
  - `residual_mean -0.002337`
  - `sharpness +0.000883`
  - `psnr_hr +0.306722`
  - `residual_mean_hr -0.001824`
  - `sharpness_hr +0.000347`
- CLI 相比手工 continuation 的差值几乎为 0:
  - `delta psnr = -1.2637e-05`
  - `delta residual_mean = -5.2154e-08`
  - `delta psnr_hr = +2.0204e-05`
  - `delta residual_mean_hr = -3.7253e-09`

### 结论 2

- 已验证结论:
  - `Phase E` 在更长 `stage3sr` 末态上继续接 `stage3b`, 仍然有稳定收益。
  - 新增的 `start_stage=stage3b --resume` workflow 已经和手工 continuation 对齐, 可以作为正式实验入口使用。
- 当前真正还没决定的只剩下:
  - auto gate 是否应该放宽
  - `stage3b` 内部超参数该怎么校准

## [2026-03-10 11:34:00 UTC] Phase E calibration 预分析

### 现象

- 基线 continuation 目录:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
- `metrics_stage3b.json` 尾部 8 个点持续单调改善:
  - `psnr 24.5487 -> 24.6336`
  - `residual_mean 0.032988 -> 0.032581`
  - `psnr_hr 21.8726 -> 21.9265`
  - `residual_mean_hr 0.048153 -> 0.047799`
- 这说明当前最先值得验证的不是立刻放松 regularizer, 而是确认 `32 iter` 是否只是预算过短。

### 当前假设

- 主假设:
  - `stage3b` 在 `iter=32` 时仍未收敛, 因此先把 `iters_stage3b` 提到 `64` 是最干净的第一刀。
- 备选解释:
  - 即便曲线仍在涨, 真正限制可能是 `means_delta_cap_stage3b=0.01` 或 anchor / rotation reg 偏强。
- 推翻主假设的证据:
  - 如果 `iter=64` 相比 `iter=32` 只带来极小增益, 或中后段开始恶化, 那就说明预算不是主要瓶颈。

## [2026-03-10 11:45:00 UTC] Phase E 第一轮 continuation calibration: `iters_stage3b=64`

### 现象

- 新目录:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter64_20260310`
- 配置口径:
  - 正式 `--start-stage stage3b --resume`
  - `--target-subsample 8`
  - 仅把 `iters_stage3b` 从 `32` 改到 `64`
  - 其余保持:
    - `lambda_means_anchor_stage3b = 0.02`
    - `lambda_rotation_reg_stage3b = 0.02`
    - `means_delta_cap_stage3b = 0.01`
- 最终 `diagnostics.json`:
  - `phase_reached = stage3b`
  - `stopped_reason = metrics_plateau`
  - `warm_start_stage3b = true`
  - `psnr = 24.962974696829484`
  - `residual_mean = 0.031111249700188637`
  - `psnr_hr = 22.124704905272846`
  - `residual_mean_hr = 0.046562712639570236`

### 静态 / 动态证据

- 相比 `iter32` continuation:
  - `psnr +0.329351070733`
  - `residual_mean -0.001469338313`
  - `psnr_hr +0.190583159265`
  - `residual_mean_hr -0.001184377819`
- 相比 auto gate 停在 `stage3sr`:
  - `psnr +0.808460816901`
  - `residual_mean -0.003806324676`
  - `psnr_hr +0.497325559677`
  - `residual_mean_hr -0.003008514643`
- `metrics_stage3b.json` 尾部 5 个点仍连续改善:
  - `psnr 24.9266 -> 24.9630`
  - `residual_mean 0.031259 -> 0.031111`
  - `psnr_hr 22.1017 -> 22.1244`
  - `residual_mean_hr 0.046685 -> 0.046543`

### 结论

- 已验证结论:
  - `Phase E` 的 continuation 不只是“多跑一点也许更好”, 而是已经证明 `iters_stage3b=64` 明显优于 `iter32`。
  - 这说明第一轮 calibration 已经回答了一个核心问题:
    - `32 iter` 不是一个充分预算。
- 当前候选假设:
  - 预算仍可能是主要瓶颈之一。
- 最强备选解释:
  - 真正卡住的也可能不是 iter 本身, 而是 `means_delta_cap_stage3b=0.01` 或两个 regularizer 偏强。
- 下一步最小可证伪实验:
  - 保持 `iters_stage3b>=64`, 再扫 `means_delta_cap_stage3b` 与 `lambda_*_stage3b`。

## [2026-03-14 00:00:00 UTC] 主题: `diffusion_output_generated_my` 生成链路与相机资产语义

### 现象

- `README.md` 里给出了精确命令:
  - `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py`
  - `--video_save_folder assets/demo/static/diffusion_output_generated_my`
- 实际磁盘目录 `/workspace/lyra/assets/demo/static/diffusion_output_generated_my` 的布局是:
  - `0..5/pose/dj-style.npz`
  - `0..5/intrinsics/dj-style.npz`
  - `0..5/rgb/dj-style.mp4`
  - `0..5/latent/dj-style.pkl`

### 已验证事实

- 目录布局由 `gen3c_single_image_sdg.py` 的 `_build_output_paths()` 统一生成:
  - `pose -> <video_save_folder>/pose/<clip>.npz`
  - `intrinsics -> <video_save_folder>/intrinsics/<clip>.npz`
  - `rgb -> <video_save_folder>/rgb/<clip>.mp4`
- `demo_multi_trajectory()` 会把根目录进一步改成 `<video_save_folder>/<traj_idx>`:
  - `0 -> left`
  - `1 -> right`
  - `2 -> up`
  - `3 -> zoom_out`
  - `4 -> zoom_in`
  - `5 -> clockwise`
- `pose.npz` 的写出内容不是 `w2c`, 而是 `generated_w2cs.inverse()` 之后的 `generated_c2ws`.
- `generate_camera_trajectory()` 的返回值文档明确写的是 `generated_w2cs`.
- `intrinsics.npz` 的写出内容是 `3x3 K` 里的:
  - `K[0,0]`
  - `K[1,1]`
  - `K[0,2]`
  - `K[1,2]`
- 下游 `DataField.CAMERA_INTRINSICS` 的契约明确是 OpenCV pinhole `[fx, fy, cx, cy]`.
- 下游 provider 的契约明确是:
  - 原始 `c2w` -> `cam_view = inverse(c2w).T`
- `refinement_v2/data_loader.py` 也明确写了:
  - 当前真实资产里的外部 `pose.npz` 保存的是 raw pose / c2w
  - 只有当 key 显式是 `cam_view` 时才不做转换
- 实际资产 `dj-style.npz` 的 key 只有 `data` 和 `inds`, 不含 `cam_view`.
- 实际资产 `rgb/dj-style.mp4` 经 `ffprobe` 验证是:
  - `width=1280`
  - `height=704`
  - `nb_frames=121`
  - `fps=24`
- 实际资产 `intrinsics/dj-style.npz` 经读取验证:
  - shape `(121, 4)`
  - 六个目录内都是同一组四元组
  - 第一帧数值是 `[887.8512, 868.12115, 640.0, 352.0]`
  - 其中 `cx=640 = width/2`, `cy=352 = height/2`
- 实际资产 `pose/dj-style.npz` 经读取验证:
  - shape `(121, 4, 4)`
  - `inds = [0..120]`
  - 首帧是单位矩阵
  - 末帧旋转子矩阵行列式为 `1.0`
- 实际资产位移趋势与 `demo_multi_trajectory()` 映射一致:
  - `0` 最后一帧 `x<0`
  - `1` 最后一帧 `x>0`
  - `2` 最后一帧 `y<0`
  - `3` 最后一帧 `z<0`
  - `4` 最后一帧 `z>0`
  - `5` 中段有位移、首尾回到原点, 符合 orbit/clockwise

### 当前结论

- `/workspace/lyra/assets/demo/static/diffusion_output_generated_my` 这类目录的静态路径, 是由 `gen3c_single_image_sdg.py` 生成的, 不是 `sample.py` 写出来的。
- `pose/*.npz` 当前真实资产语义可判定为 `camera-to-world (c2w)`.
- `intrinsics/*.npz` 当前真实资产语义可判定为 OpenCV pinhole `[fx, fy, cx, cy]`.
- `0..5` 子目录不是“样本编号”, 而是六条固定相机轨迹的 `traj_idx`, downstream 再把它们当作 multi-view 的 `view_id`.

### 仍属推断的部分

- `clockwise` 目录里的完整旋转方向是“从观察者视角顺时针”还是“从世界坐标绕物体正向转一圈”, 单靠目录名和中间位移还不能完全证明。
- 但它确实是 `camera_utils.generate_camera_trajectory(... trajectory_type=\"clockwise\")` 生成的 orbit/spiral 轨迹, 这一点已被代码证据确认。

## [2026-03-14 16:17:47 UTC] 主题: `gen3c_*_sdg.py` 输出文件名过长

### 现象

- 用户执行 `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py` 时, 会生成形如:
  - `xhc_in the style of Makoto Shinkai,注意镜头移动时候,镜头光斑,灯光光影的正常,不要贴在墙上.mp4`
- 这个文件名同时存在:
  - 很长
  - 含空格
  - 含逗号等标点
- 同一个 `clip_name` 还会被复用于:
  - `rgb/*.mp4`
  - `pose/*.npz`
  - `intrinsics/*.npz`
  - `latent/*.pkl`

### 静态证据

- `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py:148` 的 `_build_clip_name(...)` 当前逻辑是:
  - 先取输入图片 stem
  - 再把 `prompt` 原样拼接到 `clip_name`
  - batch 模式再追加 `index`
- `cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py:148` 也有同样逻辑。
- 对照脚本 `cosmos_predict1/diffusion/inference/gen3c_single_image.py:576` 使用的是:
  - `args.video_save_name`
  - 或 batch index
  - 不会把整段 prompt 直接拼进 `.mp4` 名字

### 当前主假设

- 这是历史兼容逻辑留下来的命名策略。
- 它想解决的是“同一个输入图,不同 prompt 不能重名”。
- 但做法过于直接,把整段 prompt 当文件名的一部分,导致可读性和路径安全性都变差。

### 最强备选解释

- 这套逻辑不只是为了区分 prompt, 还承担了 resume/断点续跑兼容旧产物的职责。
- 如果直接把命名改成短名, 有可能让旧目录里的产物在预扫描时“看起来像不存在”, 从而重复生成。

### 验证计划

- 在轻量模块 `inference_utils.py` 中收拢“安全短名”生成逻辑。
- 在两个 `*_sdg.py` 中同时替换, 避免同类脚本行为分叉。
- 给新命名补最小单元测试。
- 若发现旧命名确实参与 resume, 则保留 legacy 名称回退检查, 避免破坏历史产物复用。

### 验证结果

- 上述“命名 helper 直接放在 `inference_utils.py` 并跑测试”的子假设不成立。
- 动态证据:
  - `python3 -m pytest tests/test_inference_output_naming.py -q`
  - 首轮报错:
    - `ModuleNotFoundError: No module named 'imageio'`
  - 说明纯命名测试被重推理依赖拖住了。
- 因此已回滚该做法,改成:
  - 新建轻量模块 `cosmos_predict1/diffusion/inference/output_naming.py`
  - 脚本改为从该模块导入命名 helper
  - 测试也只 import 轻量模块

### 最终结论

- 已验证根因不是 `save_video()` 或 mp4 编码层。
- 真正导致长文件名的代码路径是:
  - `gen3c_single_image_sdg.py` / `gen3c_dynamic_sdg.py`
  - 旧 `_build_clip_name(...)` 把整段 prompt 原样拼进 `clip_name`
- 最终修复策略是:
  - 新生成默认走 `输入 stem + 短 prompt hash`
  - 显式 `--video_save_name` 继续优先
  - 旧长文件名产物若已存在, 仍能被 resume 逻辑识别并复用

### 本轮关键验证命令与输出

- 语法检查:
  - `python3 -m py_compile cosmos_predict1/diffusion/inference/output_naming.py cosmos_predict1/diffusion/inference/inference_utils.py cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py tests/test_inference_output_naming.py`
  - 结果: 通过
- 定向测试:
  - `python3 -m pytest tests/test_inference_output_naming.py -q`
  - 结果: `6 passed in 0.02s`
- 用用户示例 prompt 实际演算:
  - 输出 `clip_name = xhc_97e474c6`
  - 目标视频路径 `rgb/xhc_97e474c6.mp4`
