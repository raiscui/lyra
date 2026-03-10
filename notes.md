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
