# Long-LRM Style Post Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

> **Status Correction (2026-03-06):** 这份文档描述的是 `joint_refinement_camera_gaussians_v2` 已完成主线上的增量增强,不是第二套并行程序路径。后续实现必须继续复用同一个入口 `scripts/refine_robust_v2.py` 与同一个包 `src/refinement_v2/`.

**Goal:** 基于 `specs/long_lrm_style_post_refinement.md`, 在已经完成的 `joint_refinement_camera_gaussians_v2` 主线上继续增强: `sample.py` 继续做 feed-forward 初始化, `scripts/refine_robust_v2.py` 继续作为唯一后处理入口, 在现有 `Stage 2A / Stage 2B / Phase 3 / Phase 4` 体系之上吸收 `Long-LRM` 风格的 post-prediction optimization, 并通过 `SplatSuRe-style selective SR` 选择性吸收超分后的对等视频高频监督,再用 `Mip-inspired sampling-aware smoothing` 约束不受支持的高频细节.

**Architecture:** 沿用现有 `refinement_v2` 骨架, 不另开新入口, 也不再定义第二套主流程. 更准确地说:
- 外部 CLI 仍然是 `scripts/refine_robust_v2.py`
- 运行包仍然是 `src/refinement_v2/`
- `Long-LRM` / `SplatSuRe` / `Mip-Splatting` 的吸收方式,是作为现有 `stage2a` 主线内部的增强子阶段与增强 loss

第一版增强执行顺序固定为: `SceneBundle` 装配与对齐 -> baseline render / diagnostics -> `W_robust` 构造 -> `stage2a` 内部先做 `Stage 3A native cleanup` -> 可选 opacity / pruning -> `gaussian_fidelity_score + W_sr_select` -> `Stage 3SR selective SR patch supervision`(`W_final_sr = W_robust * W_sr_select`) -> `L_sampling_smooth` 约束 -> 再按现有 gating 进入可选 `Stage 3B limited geometry` -> state / export / resume. `sample.py` 不改主链, V1 不做 pose optimization, 也不引入 Long-LRM 的长序列 token budget 控制.

**Tech Stack:** Python 3.10, PyTorch, existing Lyra provider / renderer (`src/models/data/provider.py`, `src/rendering/gs_deferred.py`, `src/rendering/gs_deferred_patch.py`), OmegaConf / YAML, pathlib / json, pytest.

---

## Current Status On V2 Mainline

这份计划最初是按“从 0 到 1”写的.
但按当前代码现实, 其中很大一部分已经不是未来任务了, 而是已落地状态.

当前已经明确成立的事实:

- 唯一入口仍是 `scripts/refine_robust_v2.py`
- 唯一实现包仍是 `src/refinement_v2/`
- `joint_refinement_camera_gaussians_v2` 已经完成
- `Long-LRM style post refinement` 现在应理解为:
  - 建立在 v2 主线上的增强子计划
  - 不是第二套程序

当前代码里已经具备:

- `Phase 0`
- `Phase 1`
- `Stage 2A`
- `Stage 2A` 内部的:
  - `Stage 3A native cleanup`
  - `Phase 3S`
  - `Stage 3SR`
- `Stage 2B`
- `Phase 3`
- `Phase 4`
- pruning
- patch supervision
- selective SR
- `L_sampling_smooth`
- external reference contract
- state / resume / diagnostics / before-after 导出

当前验证状态:

- `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
- 结果是 `82 passed`

当前在主线里新增并已落地的能力:

- direct file inputs v1
  - `--pose-path`
  - `--intrinsics-path`
  - `--rgb-path`
- `build_scene_bundle(...)` 现在既能走 provider 模式
- 也能直接从本地 `pose / intrinsics / rgb` 文件组装 `SceneBundle`
- full-view root inputs v1
  - `--scene-stem`
  - `--view-ids`
  - `--pose-root`
  - `--intrinsics-root`
  - `--rgb-root`
  - `--reference-root`
- `build_scene_bundle(...)` 现在也能从显式 multi-view roots 组装一个 full-view `SceneBundle`
- 组装策略是:
  - 保留 runner 当前 5D 输入
  - 在 loader 层按 `view-major` 展平 observation 轴
- `SceneBundle` 现在会保留:
  - `view_ids`

因此, 这份文档接下来更适合承担两件事:

1. 说明哪些增强已经并入 v2 主线
2. 说明下一轮真正还值得继续的 continuation tasks 是什么

## Reader Guide / Current Backlog Snapshot

如果你现在是为了判断"还有什么没完成"而来, 推荐按下面顺序阅读这份文档:

1. 先看 `Current Continuation Order`
   - 这是当前真正建议继续推进的顺序
2. 再看 `Continuation Tasks`
   - 这里区分了哪些已经完成, 哪些只是还在收尾
3. 再决定是否进入 `Closest-to-SplatSuRe Track`
   - 这是高侵入二期路线, 不是当前默认 backlog
4. 最后如果只是想追溯历史, 再看 `Historical 0-to-1 Tasks`
   - 那一段是归档, 不是今天的待办

截至 `2026-03-09`, 当前更准确的 backlog 口径是:

- 最明确仍未彻底收口:
  - `Continuation Task C`
- 已落地, 但还有继续优化空间:
  - `Continuation Task B`
  - `Continuation Task D`
  - `Continuation Task A`
- 高侵入未来路线:
  - `Closest-to-SplatSuRe Track`
- 不应再当成当前 backlog:
  - `Task 1 ~ Task 10`
- 当前已弃用, 不纳入这份计划的 continuation backlog:
  - `depth anchor`

## Execution Readiness / Preflight Gate

继续执行这条线时, 需要先区分两类任务:

1. 不依赖重新跑 `sample.py` 的任务
2. 依赖重新生成 baseline `.ply` 的任务

当前仓库代码已经允许第一类任务继续推进.
但第二类任务在这台 `sm_61` 机器上存在明确硬阻塞:

- `sample.py` 当前仍是 `Mamba + Triton` 路径
- 本机实际 GPU 是 `Tesla P40 (sm_61)`
- 真实最小复现已经确认:
  - `bf16` 会触发 `ptxas ... requires sm_80+`
  - `fp16 / fp32` 也会触发 `no kernel image is available for execution on the device`

因此当前执行门槛应明确为:

- 如果 continuation task 需要重新跑 `sample.py`
  - 需要 `sm_80+` 级别 GPU
  - 或者需要已有 baseline `.ply`
  - 或者未来补出非 Triton 的 Mamba fallback
- 如果 continuation task 不需要重新跑 `sample.py`
  - 则可以继续在当前代码和当前机器上推进

## Continuation Tasks

下面这些才是当前还值得继续推进的增强项.
它们都建立在现有 v2 主线之上.

### Continuation Task A: 用真实外部 SR reference 做一轮系统验证

**Status (2026-03-07 最新):**

- 已完成真实系统验证
- 当前已补出 `scripts/run_flashvsr_reference.py`,并在 48G 主机上真实跑通:
  - `FlashVSR-Pro full mode`
  - 输入:
    - `assets/demo/static/diffusion_output_generated/3/rgb/00172.mp4`
  - 输出:
    - `outputs/flashvsr_reference/full_scale2x/3/rgb/00172.mp4`
- 已导出排查证据:
  - `native_frames/`
  - `sr_frames/`
  - `compare_frames/`
- 补充的 full-view 验证也已完成:
  - `00172` 的 6 路 `FlashVSR` reference 都已生成
  - 输出位于:
    - `outputs/flashvsr_reference/full_scale2x/<view>/rgb/00172.mp4`
  - full-view SR smoke 已真实跑到:
    - `phase_reached = stage3sr`
    - `target_subsample = 16`
    - `48 observations`
- 已完成真实 `SR vs native` 单变量对比:
  - `native_stage3sr`
    - `PSNR = 28.5179`
    - `residual_mean = 0.018922`
    - `sharpness = 0.005127`
  - `sr_stage3sr`
    - `PSNR = 28.4424`
    - `residual_mean = 0.019096`
    - `sharpness = 0.005211`
- 当前结论:
  - external SR reference 已经被 `Phase 3S / Stage 3SR` 真正消费
  - 但在当前默认参数下,它还没有成为当前默认最优路线
  - native 在 `PSNR / residual_mean` 上略优
  - SR 在 `sharpness` 上略优

**Why:**

- 这一步原本是为了把“代码闭环成立”推进到“真实 reference 证据成立”
- 当前这个目标已经完成
- 后续如果还想继续做 Task A,重点就不再是“有没有接上”
- 而是:
  - selective SR 的权重设计能否继续优化
  - external SR 的优势能否被更稳定地提取出来

**Focus(后续若继续):**

- 不再重复证明“FlashVSR 能接进来”
- 而是集中做:
  - `W_sr_select`
  - patch loss 配比
  - `L_sampling_smooth`
  - pruning / opacity 对 selective SR 的耦合影响
- 目标是让 SR 路线真正超过 native baseline,而不是只在 `sharpness` 上略有优势

**Done when(当前轮):**

- 已能说清:
  - 问题不在 `FlashVSR` 没接上
  - 也不是 direct-input `cam_view` 契约没修好
  - 当前更像是 selective SR 默认超参数还没有把外部 SR 价值真正吃满

### Continuation Task B: 打通 full-view 联合优化的 root inputs

**Status (2026-03-07 最新):**

- 第一版已经完成并做过真实资产验证
- 当前同一个 `scripts/refine_robust_v2.py` 入口已经支持:
  - `--scene-stem`
  - `--view-ids`
  - `--pose-root`
  - `--intrinsics-root`
  - `--rgb-root`
  - `--reference-root`
- 当前 `build_scene_bundle(...)` 已支持:
  - single-view direct file inputs
  - full-view native root inputs
  - full-view external SR reference root inputs
- 实现策略已经定住:
  - 不改 runner 的 5D 输入面
  - 在 loader 层做 full-view bundle flatten
  - `SceneBundle` 额外保留 `view_ids`
- 真实 full-view 验证结果:
  - native `phase0` dry-run 通过
  - native `Stage 2A` smoke 成功
  - SR `Stage 3SR` smoke 成功
- 当前 48G 主机上的推荐 observation 体量:
  - `target_subsample = 16`
  - 即 `8 frames/view * 6 views = 48 observations`
- 已确认的边界:
  - `target_subsample = 8`
  - 即 `96 observations`
  - 会在 full-view `Stage 2A` 上 OOM

**Status (2026-03-09 refresh):**

- 命令和文档的第一轮收口已经完成
- 当前已经明确写清:
  - full-view native smoke 是正式 baseline 起点
  - full-view SR smoke 是同 observation 预算下的增强分支
  - 当前 48G 主机只推荐 `target_subsample = 16`
  - 不建议在 baseline 还没固定前直接混入 `Stage 2B`
- 因此 `Continuation Task B` 当前剩余内容, 已不再是"入口没打通"
- 更准确地说, 剩下的是:
  - 更长 smoke
  - 在当前 `48 observations` 档位下继续比较 native vs SR
  - 需要时再衔接 `Stage 2B`

**Why:**

- 用户当前真正要的不是“每个 view 单独评估”
- 而是所有视频一起参与, 联合优化一个 gaussian scene
- 在这个前提下,full-view joint bundle 比继续单 view 调参更优先

**Focus(剩余工作):**

- 把 full-view 基线文档和命令收口到统一口径
- 在当前 `48 observations` 档位上继续做:
  - native vs SR 的更长 smoke
  - 需要时再接 `Stage 2B`
- 更大 observation 密度留到:
  - A100
  - 或多卡

**Done when(当前轮):**

- 同一条 v2 主线已经能走:
  - provider 模式
  - direct file inputs 模式
  - full-view root mode
- 当前 full-view 推荐基线已经明确写成:
  - `target_subsample = 16`
  - `48 observations`

### Continuation Task C: 收敛 stage 命名与使用文档

**Status (2026-03-09 refresh):**

- 仍未正式收口
- 当前最需要补的是:
  - 对外命名统一
  - usage 示例统一
  - diagnostics / 产物说明统一
- 这也是当前 continuation tasks 里最明确的开放项

**Why:**

- 代码里现在已经有:
  - `stage2a`
  - `stage3a`
  - `phase3s`
  - `stage3sr`
- 但对外文档如果不及时收敛, 很容易又被理解成第二条程序路径

**Focus:**

- 统一对外解释:
  - `stage2a` 是主线阶段名
  - `stage3a / phase3s / stage3sr` 是 `stage2a` 内部增强子阶段
- 同步更新:
  - usage 文档
  - diagnostics 说明
  - 产物说明

**Done when:**

- 使用者不会再把 selective SR 误解成新的独立程序

### Continuation Task D: 调 `Stage 2B` 和 selective SR 的衔接策略

**Status (2026-03-07 最新):**

- 当前默认衔接策略已经通过真实运行基本定住
- 已完成:
  - `native_stage3sr -> stage2b`
  - `sr_stage3sr -> stage2b`
  两条真实续跑
- 同时修复了真实 warm-start workflow 中的 resume 断裂:
  - `.ply` 导出会过滤低 opacity 高斯
  - `state/latest.pt` 保留全量高斯
  - `restore_latest_state()` 现已在数量不一致时按 state tensor 重建 `GaussianAdapter`

**Why:**

- 代码上虽然早就能“进入 `Stage 2B`”
- 但直到这轮真实对比完成前,还不能回答:
  - 当前正式主线到底该不该继续 `Stage 2B`
  - external SR 接上 `Stage 2B` 后会不会反超 native

**Focus:**

- 评估:
  - 哪些 case 应该止步在 `Stage 3SR`
  - 哪些 case 值得继续放开 `Stage 2B`
- 给出:
  - 推荐 gate 指标
  - 推荐 warm-start 方式
  - 推荐默认超参数

**Current Result:**

- `native_stage2b`
  - `PSNR = 30.7697`
  - `residual_mean = 0.014952`
  - `sharpness = 0.006509`
- `sr_stage2b`
  - `PSNR = 30.3745`
  - `residual_mean = 0.015645`
  - `sharpness = 0.006647`
- 当前结论:
  - `Stage 2B` 的收益明显大于当前 `native vs SR reference` 的差异
  - `native baseline -> Stage 2B` 是当前正式推荐主线
  - `SR -> Stage 2B` 仍可保留为锐度优先的可选分支
- `2026-03-09 refresh`:
  - 在 A800 / full-view / `target_subsample=8` / external SR 口径下:
    - `stage3sr -> stage2b (run2)`
      - `PSNR = 25.4044`
      - `residual_mean = 0.02871`
      - `sharpness = 0.004525`
    - 在同一 SR 分支继续追加一段 `Stage 2B` 后:
      - `stage2b -> stage2b (run3)`
      - `PSNR = 25.7353`
      - `residual_mean = 0.02742`
      - `sharpness = 0.004948`
  - 这说明:
    - 当前 SR 分支继续追加 `Stage 2B` 仍有真实收益
    - 但即使显式打开 `Phase 3 / Phase 4` 的开关, 当前 run 仍停在 `stage2b`
    - 原因不是流程失效, 而是 gate 已经变成:
      - `need_geometry = false`
      - `local_overlap_persistent = false`
      - `global_shift_detected = false`
    - 因此 `Phase 3 / Phase 4` 在当前实现里更像“问题兜底阶段”,不是正常健康 run 的默认后续主线

**Done when(当前轮):**

- `Stage 3SR -> Stage 2B` 的切换已经不再靠人工直觉
- 当前正式推荐 workflow 已可明确写成:
  - `native baseline -> Stage 2B`

## Historical 0-to-1 Tasks (Archive Only, Not Current Backlog)

下面的 `Task 1 ~ Task 10` 保留为这份文档最初形成时的历史任务拆解.
它们有的已经落地, 有的已经被当前代码状态覆盖.
保留它们是为了追溯思路, 不是为了把后续工作重新理解成“另起一条路线”.

## Closest-to-SplatSuRe Track

如果后续明确选择“最接近 SplatSuRe 的版本”, 当前建议不要继续在现有 `single-hotspot patch SR` 上小修小补.
更接近论文原意的实现,应该一次性把下面这几件事补齐:

### Track Goal

把当前的:

- native render 主输出
- selective SR patch 辅助项

推进成更接近 SplatSuRe 的:

- HR render 主输出
- 全图 LR consistency
- 全图 selective SR weighted supervision

### Gap Summary

当前和 SplatSuRe 的关键差异有 6 个:

1. 当前最终 render 仍是 native 分辨率,不是 HR 主输出
2. 当前 SR supervision 只落在每视角单个 hotspot patch
3. 当前 `sr_selection_map` 动态范围偏弱,没有像官方实现那样形成强监督区域
4. 当前 `W_final_sr = W_robust * W_sr_select` 过于保守
5. 当前 `Stage 3SR` 不动 geometry,也不和 densify 主循环耦合
6. 当前 fidelity score 还是 render-meta 代理,还不是论文中的跨视角 `min/max radii ratio`

### Phase A: 真正复刻 Gaussian Fidelity Score

Status (2026-03-10 refresh):

- 已完成第一轮最小落地:
  - `phase3s` 现在会导出跨视图 fidelity diagnostics:
    - `r_min`
    - `r_max`
    - `rho`
    - `num_times_seen`
    - `argmax_view`
    - `max_view_mask`
  - `gaussian_fidelity_histogram.json` 现在额外包含:
    - `rho_*`
    - `num_times_seen_*`
    - `max_view_counts`
  - `build_sr_selection_weight()` 已改成优先把低 fidelity Gaussian 只投到自己的 `max_view_mask`
- 这意味着:
  - 当前 `Phase A` 的“fidelity / max-view mask 做真”已经不再只是文档目标
  - 代码和测试里已经有第一轮可运行闭环
- 仍未完成:
  - 已补第一轮 full-view `sub8` 实验:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_sub8_iter20_20260310`
    - 相对旧 baseline:
      - `PSNR +0.00424`
      - `residual_mean -0.0000945`
      - `sharpness -0.00000430`
    - 这说明新 `Phase A` 已显著改变 selection 分布, 但最终收益目前仍接近持平
  - 已补最小 fidelity calibration:
    - CLI 已暴露:
      - `--fidelity-ratio-threshold`
      - `--fidelity-sigmoid-k`
      - `--fidelity-min-views`
      - `--fidelity-opacity-threshold`
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_thr1p1_sub8_iter20_20260310`
    - 仅把 `fidelity_ratio_threshold: 1.5 -> 1.1`
    - 相对 `Phase A` rerun:
      - `PSNR -0.000586`
      - `residual_mean +0.0000147`
      - `sharpness -0.000000133`
      - `selection_mean +0.03125`
    - 这说明:
      - selection coverage 确实被明显放大了
      - 但最终指标仍没有超过当前 `Phase A`
      - 当前瓶颈已经不再只是 fidelity routing 强度
  - `Phase B ~ E` 仍然保持待实现

目标:

- 不再只依赖单轮 render meta 的代理量
- 而是像 SplatSuRe 一样,在全部训练视角上统计:
  - `r_min`
  - `r_max`
  - `argmax_view`
  - `num_times_seen`

要点:

- fidelity score 应直接基于:
  - `rho = r_max / r_min`
- 至少保留:
  - `ratio_threshold`
  - 平滑参数 `k`
  - `num_times_seen < 3 -> score = 0`
- 对每个训练视角额外记录:
  - 哪些高斯的最大半径恰好出现在该视角

完成标志:

- 当前 `phase3s` 不再只是“第一版代理实现”
- 而是能生成真正接近 SplatSuRe 的:
  - per-Gaussian fidelity
  - per-view max-view mask

### Phase B: 从单 patch 升级到全图 weighted SR supervision

Status (2026-03-10 refresh):

- 已完成第一轮最小落地:
  - 新增 `--sr-patches-per-view`
  - 当前 `Stage 3SR` 已不再被迫只吃单热点 patch
  - 会按 reference priority 选多个 patch set, 再逐个渲染并平均 patch loss
- 已完成两轮 full-view `sub8` 验证:
  1. `multi4`
     - `outputs/refine_v2/full_view_sr_stage3sr_phaseB_multi4_sub8_iter20_20260310`
     - `phase3s` 能完成
     - 但 `stage3sr` 在 reference patch render 时 OOM
  2. `multi2`
     - `outputs/refine_v2/full_view_sr_stage3sr_phaseB_multi2_sub8_iter20_20260310`
     - 可以完整跑完
     - 相对旧 baseline:
       - `PSNR +0.00112`
       - `residual_mean -0.0000843`
       - `sharpness -0.00000244`
     - 相对 `Phase A` rerun:
       - 指标基本持平, 且略弱于 `Phase A`
- 这说明:
  - `Phase B` 的多 patch 机制已经代码落地
  - 但当前最小版 `multi2` 还没有把收益明显拉高
  - `multi4` 在当前 `patch-size=256` 口径下又太重

目标:

- 去掉“每视角只取一个 residual hotspot patch”的限制
- 改成整张 HR reference 图上的逐像素加权监督

要点:

- 不再由:
  - `sample_patch_windows()`
  决定 SR 的有效区域
- 而是直接使用:
  - `W_sr_select`
  在整张 HR 图上形成 loss
- 如果显存不允许一次整图:
  - 也应该是“按全图权重图切多个 patch”
  - 不是单个 patch

完成标志:

- SR loss 的覆盖范围由选择图决定
- 而不是由单个热点裁剪窗口决定

### Phase C: 引入真正的 LR-SR 联合目标

Status (2026-03-10 refresh):

- 当前方向已经重新澄清:
  - 不是继续维持“native LR 主损失 + SR patch 辅助项”
  - 而是让 6 路 SR 视频正式升格为新的主监督
  - native LR 只保留为下采样一致性约束
- 这背后的直接原因是:
  - 当前代码已确认仍在同时使用:
    - `gt_images` 作为 native LR 主 RGB 监督
    - `reference_images` 作为 SR reference 监督
  - 同时第一轮 fidelity calibration 也说明:
    - 即便明显放大 selection coverage
    - 最终指标仍没有超过当前 `Phase A`
  - 因此当前更值得改的是 supervision objective, 而不只是继续调 routing 强度
- 当前不采用的版本是:
  - 直接把 native LR 全部删除
  - 改成纯 `HR-only supervision`
- 原因是:
  - SR 视频虽然和原视频共享同一套相机参数
  - 但它仍然是后验超分结果
  - 如果完全去掉 native LR 约束, 更容易把 SR hallucination 直接写进 3D
- 2026-03-10 implementation update:
  - `Phase C.1 ~ C.4` 的最小工程闭环已经落地到代码
  - `Stage 3SR` 现在支持 `full_frame_hr` 监督模式
  - 真实 full-view external SR + `sub8` 最小 smoke 已通过:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_streamshard_20260310`
  - 随后的第一轮更长对照也已完成:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr1_sub8_iter8_20260310`
    - 结果达到:
      - `psnr = 22.2530`
      - `residual_mean = 0.044018`
      - `psnr_hr = 20.2697`
  - 第二轮更长对照也已完成:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter20_20260310`
    - 当前最佳结果达到:
      - `psnr = 23.5465`
      - `residual_mean = 0.037822`
      - `sharpness = 0.002699`
      - `psnr_hr = 21.2458`
      - `residual_mean_hr = 0.051848`
    - 与 `Phase A iter20` 相比:
      - native `psnr` 差距缩到 `-0.4545`
      - native `residual_mean` 差距缩到 `+0.003393`
    - 这说明:
      - `Phase C` 的长程收益已经被真实验证
      - 当前再继续停留在 `1 iter` sweep 的价值更低
      - 当时更值得先把 `Phase D` 补完
  - 第三轮更长对照也已完成:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter32_20260310`
    - 当前最佳结果达到:
      - `psnr = 24.1546`
      - `residual_mean = 0.034917`
      - `sharpness = 0.003191`
      - `psnr_hr = 21.6160`
      - `residual_mean_hr = 0.049654`
    - 与 `Phase C iter20` 相比:
      - `psnr +0.6081`
      - `residual_mean -0.002905`
      - `sharpness +0.000493`
      - `psnr_hr +0.3702`
      - `residual_mean_hr -0.002194`
    - 与 `Phase A iter20` 相比:
      - native `psnr` 已反超 `+0.1536`
      - native `sharpness` 已反超 `+0.000236`
      - native `residual_mean` 仍略高 `+0.000488`
    - 这说明:
      - `Phase C` 已经跨过“是否成立”的阶段
      - `Phase D` 也已经完成, 因而当前主线问题不再是“HR 导出是否缺失”
      - 后续真正值得继续压的是 native `residual_mean` 能否也一起超过 `Phase A`
  - 额外确认了一条重要工程规律:
    - 只做 `serial render` 不够
    - 必须进一步做 `stream-sharded loss/backward`, 否则后续 full concat 仍会把峰值显存拉回来
  - 同时也确认了当前实验策略上的一条规律:
    - `lambda_hr_rgb=8/16/32` 在 `1 iter` smoke 上几乎无差异
    - 后续参数实验更值得优先拉长 iter, 或开始下调 `lambda_lr_consistency`

目标:

- 让 `Stage 3SR` 从当前的“native 主损失 + SR patch 辅助”升级成:
  - HR render 对 6 路 SR 视频做主监督
  - HR render 下采样后,再和 native LR 视频做全图一致性约束
- 换句话说:
  - SR 6 视频成为新的“主 GT”
  - native LR 不再是并行主目标,而是 anti-hallucination consistency

要点:

- 新目标应更接近:
  - `L = gamma * L_hr_sr + (1 - gamma) * L_lr_consistency`
- 这里的 `L_lr_consistency` 语义已经变了:
  - 不再是当前那种 native full-frame 主损失的简单延续
  - 而是“HR 预测必须在降回 LR 后仍然对得上原视频”
- 更接近当前共识的版本是:
  1. 在 reference 尺度渲染 full-frame HR output
  2. 直接对 HR output 和 SR 6 视频做主 photometric supervision
  3. 把 HR output 下采样回 native 分辨率
  4. 再对 downsampled output 和 native LR 做全图 consistency loss
  5. selective weighting 继续保留, 但它服务的是“HR 主监督的权重分布”, 不再只是 patch 旁路
- 从工程挂点看, 这轮最小实现应优先复用:
  - 现有 `reference_images / intrinsics_ref`
  - 现有 `render_scene(...)` 与 renderer cache
  - 现有 `compute_weighted_rgb_loss(...)`
  - 然后补:
    - full-frame HR render helper
    - HR -> LR downsample consistency helper
    - 新的 phase/stage diagnostics

完成标志:

- 当前主训练目标不再是“native 主目标 + SR patch 旁路”
- 而是“SR 主监督 + LR consistency + optional selective weighting”

Current agreed implementation tasks:

1. `Phase C.1` 先补 full-frame HR render 路径
   - 让 runner 能在 reference 分辨率 / `intrinsics_ref` 下直接渲染整图
   - 不再依赖 patch scene 作为唯一 HR render 入口
2. `Phase C.2` 再补 HR -> LR consistency 路径
   - 明确下采样策略
   - 明确 native LR consistency loss 的计算位置
3. `Phase C.3` 把 `Stage 3SR` 目标重组为:
   - HR 对 SR 6 视频主监督
   - LR consistency 对 native GT
   - selective weighting 作为 HR supervision 的权重层
4. `Phase C.4` 补 diagnostics / metrics / tests
   - 至少拆开:
     - HR-space metrics
     - LR consistency metrics
     - memory / render mode diagnostics

### Phase D: 让最终输出真的成为 HR 输出

Status (2026-03-10 refresh):

- `Phase D` 的最小闭环已经落地:
  - 只要当前 scene 真的存在独立的 `reference-space`
  - runner 就会在 baseline/final 导出时额外补出:
    - `baseline_render_hr.mp4`
    - `gt_reference_hr.mp4`
    - `final_render_hr.mp4`
- `diagnostics.json` 现在会额外记录:
  - `native_hw`
  - `reference_hw`
  - `baseline_hr`
  - `final_hr`
- 对应回归已验证:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2`
  - 结果: `113 passed`
- 真实 smoke 也已成功验证:
  - `phase0` 级:
    - `outputs/refine_v2/phaseD_phase0_hr_export_smoke_20260310`
    - 已确认同时落盘:
      - native videos / snapshots
      - hr videos / snapshots
    - `diagnostics.json` 已确认包含:
      - `baseline_hr`
      - `final_hr`
      - `psnr_gain_hr`
      - `sharpness_gain_hr`
      - `residual_mean_hr_drop`
  - 更重的 `full-view + sub8 + stage2a` 级:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseD_export_smoke_sub8_20260310_final`
    - 已确认:
      - `phase_reached = stage3sr`
      - `final_render_hr.mp4` 正常落盘
      - `psnr_gain_hr = 0.9238`
      - `residual_mean_hr_drop = 0.013701`

目标:

- 最终 `final_render.mp4` 不再停留在 native 分辨率
- 而是能直接输出 reference 尺度的 HR render

要点:

- 当前 `self.scene` 仍是 native 场景
- 需要明确区分:
  - native scene
  - hr scene
- 最终指标也需要拆开:
  - native-space metrics
  - hr-space metrics

完成标志:

- 最终产物默认至少包含:
  - native render
  - hr render
- 并能直接回答:
  - 这轮优化到底有没有提升高分输出

### Phase E: 允许 SR 真正改变结构,而不只是改纹理

Status (2026-03-10 refresh):

- `Phase E` 的最小版已经落地, 后续第一轮补强与 continuation workflow 也已完成:
  - 新增 `enable_stage3b`
  - 新增 `StageController.should_enter_stage3b(...)`
  - 新增 `run_stage3b()`
  - 复用 `Phase C` 的 full-frame HR 主监督路径, 在其后允许有限 geometry 更新
- `Phase E` continuation 现在也有正式入口:
  - `start_stage=stage3b`
  - `warm_start_stage3b`
  - `--resume` + `state/latest.pt` 可直接把 `stage3sr` 末态续到 `stage3b`
- `stage3b` 专属超参数面也已经落地:
  - `iters_stage3b`
  - `lambda_means_anchor_stage3b`
  - `lambda_rotation_reg_stage3b`
  - `means_delta_cap_stage3b`
- 对应回归已验证:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2`
  - 结果: `122 passed`
- 真实 smoke 也已成功验证:
  - `outputs/refine_v2/phaseE_stage3b_smoke_sub8_20260310`
  - 已确认:
    - `phase_reached = stage3b`
    - `metrics_stage3sr.json` 与 `metrics_stage3b.json` 都存在
    - 同一条 run 内 `stage3b` 相比 `stage3sr`:
      - `psnr +0.8069`
      - `residual_mean -0.007219`
      - `psnr_hr +0.5378`
      - `residual_mean_hr -0.005412`
- 新一条专属参数 smoke 也已验证:
  - `outputs/refine_v2/phaseE_stage3b_hparams_smoke_sub8_20260310`
  - `metrics_stage3b.json` 最后一点明确记录:
    - `iters_budget = 2`
    - `lambda_means_anchor_active = 0.02`
    - `lambda_rotation_reg_active = 0.02`
    - `means_delta_cap_active = 0.01`
- 更长的 auto-gate apples-to-apples run 也已验证:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_hr32_lr0p5_sub8_iter32_20260310`
  - 这条 run 最终停在:
    - `phase_reached = stage3sr`
    - `residual_mean = 0.034917574375867844`
  - 当前解释不是“`stage3b` 没收益”, 而是:
    - 这条 run 的 `stage3sr` 已经把 overlap 压到当前 gate 以下
- 从同一条 `stage3sr` 末态继续接 `stage3b` 也已验证:
  - 手工 continuation:
    - `outputs/refine_v2/full_view_sr_stage3b_phaseE_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
  - 正式 CLI continuation:
    - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
  - 两条 continuation 都收敛到几乎同一结果:
    - `psnr ≈ 24.6336`
    - `residual_mean ≈ 0.0325806`
    - `psnr_hr ≈ 21.9341`
    - `residual_mean_hr ≈ 0.0477471`
  - 相比 gate 停在 `stage3sr` 的结果:
    - `psnr +0.4791`
    - `residual_mean -0.002337`
    - `psnr_hr +0.3067`
    - `residual_mean_hr -0.001824`
- 第一轮 continuation calibration 也已验证:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter64_20260310`
  - 做法:
    - 保持 `lambda_means_anchor_stage3b=0.02`
    - 保持 `lambda_rotation_reg_stage3b=0.02`
    - 保持 `means_delta_cap_stage3b=0.01`
    - 只把 `iters_stage3b` 从 `32` 提到 `64`
  - 最终:
    - `psnr = 24.962974696829484`
    - `residual_mean = 0.031111249700188637`
    - `psnr_hr = 22.124704905272846`
    - `residual_mean_hr = 0.046562712639570236`
  - 相比 `iter32` continuation:
    - `psnr +0.3294`
    - `residual_mean -0.001469`
    - `psnr_hr +0.1906`
    - `residual_mean_hr -0.001184`
  - 已观察到:
    - `metrics_stage3b.json` 尾部 5 个点仍持续改善
- 当前剩余限制也要明确:
  - auto gate 仍然偏保守, 还没有决定要不要调阈值
  - `32 iter` 已可确认偏短, 但 `64 iter` 之后是否继续加预算还没最终回答
  - `means_delta_cap_stage3b` 与 `lambda_*_stage3b` 还没有做正式 calibration
  - 还没有把 `stage3b` 与 densify / prune 更紧地耦合

目标:

- 让 SR 信息可以对 geometry 产生有限但真实的影响

要点:

- 当前 `stage3b` 已经做到:
  - 在 `Stage 3SR` 之后继续沿用同一套 HR 主监督 + LR consistency
  - 同时有限释放 `means / rotations`
- 当前 `stage3b` 也已经不再和 `stage2b` 绑定同一套 geometry 配置:
  - iteration budget
  - means anchor 权重
  - rotation regularizer 权重
  - means delta cap
- 更接近 SplatSuRe / 3DGS 主训练形态的后续版本, 当前至少还要评估两条路:
  1. 用正式 `start_stage=stage3b --resume` workflow 做 `Phase E` 内部 calibration
  2. 再决定 auto gate 的 `residual_mean` 阈值是否要放宽
  3. 把 SR 阶段和 densify / prune 更紧地耦合

完成标志:

- SR 信息不再只表现为轻微锐度增益
- 而能在某些结构缺失区带来更明显的可见收益

### Recommended Execution Order

如果真要做“最接近 SplatSuRe”的版本, 当前推荐顺序是:

1. Phase C: 先把目标改对
   - SR 6 视频升格为主监督
   - native LR 退到 downsample consistency
2. Phase D: 让最终导出支持 HR render
   - 否则就算 `Phase C` 生效, 最终交付物也还看不到真正 HR 输出
3. 再回头补强 Phase B
   - 如果 `Phase C` 之后仍存在 coverage 不足或显存浪费, 再重做“全图 weighted SR”实现
4. 最后再评估 Phase E
   - 是否让 SR 信息有限进入 geometry / densify
5. Phase A calibration 只保留为低优先级微调
   - 不是当前主线 blocker

### Practical Warning

这条路线的侵入性明显高于当前主线.
它不是“再调几个超参数”就能得到的.

但如果目标真的是:

- 最大限度复刻 SplatSuRe 的方法论
- 让 external SR 真正成为高分主监督

那它就是当前最正确的方向.

---

### Task 1: 对齐 CLI 与配置层到新 spec

**Files:**
- Modify: `scripts/refine_robust_v2.py`
- Modify: `src/refinement_v2/config.py`
- Test: `tests/refinement_v2/test_config.py`
- Read: `specs/long_lrm_style_post_refinement.md`

**Step 1: 先写失败测试, 锁住新 CLI 形状**

测试至少覆盖:

- `--gaussians`
- `--pose`
- `--intrinsics`
- `--reference-video`
- `--reference-mode`
- `--sr-scale`
- `--scene-index`
- `--view-id`
- `--outdir`
- `--frame-indices`
- `--target-subsample`
- `--patch-size`
- `--dry-run`
- `--resume`
- `--enable-stage3b`

建议先在 `tests/refinement_v2/test_config.py` 增加类似断言:

```python
def test_cli_maps_long_lrm_post_refinement_args() -> None:
    run_config, stage_hparams = load_effective_config_from_cli(
        [
            "--config", "configs/demo/lyra_static.yaml",
            "--gaussians", "outputs/demo/gaussians_0.ply",
            "--pose", "assets/demo/pose/demo.npz",
            "--intrinsics", "assets/demo/intrinsics/demo.npz",
            "--reference-video", "assets/demo/rgb/demo.mp4",
            "--reference-mode", "super_resolved",
            "--sr-scale", "2.0",
            "--scene-index", "0",
            "--view-id", "3",
            "--outdir", "outputs/refine_v2/demo",
            "--patch-size", "256",
            "--enable-stage3b",
        ]
    )

    assert str(run_config.gaussians_path).endswith("gaussians_0.ply")
    assert run_config.reference_mode == "super_resolved"
    assert run_config.sr_scale == 2.0
    assert run_config.enable_stage3b is True
    assert stage_hparams.patch_size == 256
```

**Step 2: 跑测试, 确认当前确实还不满足**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_config.py
```

Expected:

- 新增断言先失败
- 失败点集中在字段缺失、默认值不完整, 或 CLI 参数未注册

**Step 3: 实现最小配置层**

在 `src/refinement_v2/config.py` 至少明确这几个对象:

```python
@dataclass
class RefinementRunConfig:
    config_path: Path
    gaussians_path: Path
    pose_path: Path
    intrinsics_path: Path
    reference_video_path: Path
    reference_mode: str
    sr_scale: float
    scene_index: int
    view_id: str | None
    outdir: Path
    frame_indices: list[int] | None
    target_subsample: int | None
    dry_run: bool
    resume: bool
    enable_stage3b: bool


@dataclass
class StageHyperParams:
    patch_size: int
    weight_floor: float
    residual_tau: float
    opacity_prune_threshold: float
    prune_every: int
    means_delta_cap: float
```

并让 `scripts/refine_robust_v2.py` 只负责:

- 解析 CLI
- 加载配置
- 把 config 交给 runner

**Step 4: 再跑测试和帮助信息**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_config.py
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py --help
```

Expected:

- `test_config.py` 全部通过
- `--help` 能打印出新参数

**Step 5: 提交本任务(可选, 若按 task 落地)**

```bash
git add scripts/refine_robust_v2.py src/refinement_v2/config.py tests/refinement_v2/test_config.py
git commit -s -m "feat: align refine cli with post refinement spec"
```

**Done when:**

- 新 spec 需要的 CLI 全部能落到 `RefinementRunConfig`
- `reference_mode` / `sr_scale` / `enable_stage3b` 有明确默认值
- 明确声明 V1 不含 pose optimization 和 token budget 控制

---

### Task 2: 实现 `SceneBundle` 与 reference video 对齐层

**Files:**
- Modify: `src/refinement_v2/data_loader.py`
- Modify: `src/refinement_v2/config.py`
- Test: `tests/refinement_v2/test_data_loader.py`
- Read: `src/models/data/provider.py`

**Step 1: 先写 synthetic 单测**

至少覆盖:

- `frame_indices` 过滤
- `target_subsample` 生效
- `reference_mode=native` 时 `reference_hw == native_hw`
- `reference_mode=super_resolved` 时 `intrinsics_ref == intrinsics_native * sr_scale`
- 帧数不一致时报错
- aspect ratio / crop 不匹配时报错

建议补一个显式的失败测试:

```python
def test_super_resolved_reference_requires_scaled_intrinsics() -> None:
    bundle = build_scene_bundle(
        gaussians_path=Path("dummy.ply"),
        pose_path=pose_path,
        intrinsics_path=intrinsics_path,
        reference_video_path=reference_path,
        reference_mode="super_resolved",
        sr_scale=2.0,
        scene_index=0,
        view_id="3",
    )

    assert torch.allclose(bundle.intrinsics_ref, bundle.intrinsics_native * 2.0)
```

**Step 2: 跑测试, 确认对齐约束尚未落地**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_data_loader.py
```

Expected:

- 新增 SR / mismatch 测试先失败

**Step 3: 实现 `SceneBundle`**

建议把 `data_loader.py` 里的核心对象收紧成:

```python
@dataclass
class SceneBundle:
    gaussians_path: Path
    reference_frames: torch.Tensor
    cam_view: torch.Tensor
    intrinsics_native: torch.Tensor
    intrinsics_ref: torch.Tensor
    native_hw: tuple[int, int]
    reference_hw: tuple[int, int]
    sr_scale: float
    reference_mode: str
    frame_indices: list[int]
    scene_index: int
    view_id: str | None
```

同时实现:

- `load_reference_video(...)`
- `scale_intrinsics(...)`
- `validate_reference_alignment(...)`
- `build_scene_bundle(...)`

这里要坚持 fail-fast:

- 帧数不一致直接报错
- SR 不是整数倍率直接报错
- crop / aspect ratio 不匹配直接报错

**Step 4: 再跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_data_loader.py
```

Expected:

- `SceneBundle` 相关测试全部通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/data_loader.py src/refinement_v2/config.py tests/refinement_v2/test_data_loader.py
git commit -s -m "feat: build aligned scene bundle for post refinement"
```

**Done when:**

- runner 后面只接 `SceneBundle`, 不直接碰 provider 原始 batch
- native / SR 的 intrinsics 约束被统一固定

---

### Task 3: 打通 baseline render 与第一批 diagnostics

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Modify: `tests/refinement_v2/test_diagnostics.py`
- Create: `tests/refinement_v2/test_runner_baseline.py`
- Read: `src/rendering/gs_deferred.py`

**Step 1: 先写 baseline 层测试**

至少覆盖:

- `render_baseline()` 能返回与 `frame_indices` 对应的渲染结果
- `compute_baseline_metrics()` 输出:
  - `psnr`
  - `sharpness`
  - `opacity_lowconf_ratio`
  - `scale_tail_ratio`
- baseline 失败时 runner 不进入后续优化阶段

建议用 monkeypatch 或 synthetic renderer stub, 不要一开始依赖真实大模型.

**Step 2: 跑测试, 让 baseline contract 先红**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_diagnostics.py tests/refinement_v2/test_runner_baseline.py
```

Expected:

- baseline / diagnostics 的新 contract 先失败

**Step 3: 实现最小 baseline 路径**

在 `runner.py` 增加类似骨架:

```python
class RefinementRunner:
    def render_baseline(self, bundle: SceneBundle) -> BaselineResult:
        ...

    def run(self) -> None:
        bundle = build_scene_bundle(...)
        baseline = self.render_baseline(bundle)
        self.write_baseline_artifacts(baseline)
        if self.config.dry_run:
            return
```

在 `diagnostics.py` 实现:

- `summarize_baseline_metrics(...)`
- `summarize_opacity_stats(...)`
- `summarize_scale_stats(...)`

先把结构稳定下来, 指标值可以先依赖 synthetic / stub 输出做验证.

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_diagnostics.py tests/refinement_v2/test_runner_baseline.py
```

Expected:

- baseline contract 通过
- dry-run 有 baseline 产物, 但不会继续进入优化

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/runner.py src/refinement_v2/diagnostics.py tests/refinement_v2/test_diagnostics.py tests/refinement_v2/test_runner_baseline.py
git commit -s -m "feat: add baseline render and diagnostics"
```

**Done when:**

- `Phase 1: baseline render and diagnostics` 可独立跑通
- baseline 失败会中止流程, 不会硬进优化阶段

---

### Task 4: 落地 `W_robust` 构造器

**Files:**
- Modify: `src/refinement_v2/weight_builder.py`
- Modify: `src/refinement_v2/losses.py`
- Test: `tests/refinement_v2/test_weight_builder.py`

**Step 1: 先写失败测试**

至少覆盖:

- residual 全 0 时, 权重接近全 1
- residual 极高时, 权重下降但不低于 `weight_floor`
- quantile normalize 后结果稳定在预期范围
- EMA 第二次更新比第一次更平稳
- full-frame 和 patch tensor 的 shape 都能通过

**Step 2: 跑测试, 确认 `WeightBuilder` 还没长成 spec 里的样子**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_weight_builder.py
```

Expected:

- 新增 residual / EMA / patch 测试先失败

**Step 3: 实现 V1 最小版**

第一版只做 spec 已经收敛过的部分:

- RGB residual
- quantile normalize
- `exp(-r_norm / tau)`
- `clamp(weight_floor, 1.0)`
- EMA
- optional blur

不要在这一步把 LPIPS / DINO / feature residual 一起塞进来.

建议对外接口固定成:

```python
class WeightBuilder:
    def build_from_residual(self, residual: torch.Tensor) -> torch.Tensor: ...
    def update_ema(self, weights: torch.Tensor) -> torch.Tensor: ...
```

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_weight_builder.py
```

Expected:

- 所有 residual / weight map 单测通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/weight_builder.py src/refinement_v2/losses.py tests/refinement_v2/test_weight_builder.py
git commit -s -m "feat: add residual weight builder"
```

**Done when:**

- `Phase 2: robust residual / trust weight build` 已可独立验证
- `W_robust` 的语义已经稳定
- 后面 Stage 3A / selective SR 都能复用同一套基础权重接口

---

### Task 5: 打通 `Long-LRM` 风格的 Stage 3A appearance-first 优化主线

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/stage_controller.py`
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/losses.py`
- Test: `tests/refinement_v2/test_gaussian_adapter.py`
- Test: `tests/refinement_v2/test_losses.py`
- Create: `tests/refinement_v2/test_runner_stage3a.py`

**Step 1: 先写失败测试**

至少覆盖:

- `Stage 3A` 默认只更新:
  - opacity
  - color
  - limited scale
- `means` / `rotation` 在 Stage 3A 中被冻结
- `L_rgb_weighted` 与 `L_opacity_sparse` 会一起出现在 loss dict
- runner 在一次 stage 内的顺序固定为:
  - render
  - residual
  - weight
  - loss
  - backward
  - step

**Step 2: 跑测试, 让 Stage 3A 的 contract 先失败**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_stage3a.py
```

Expected:

- 参数冻结与 stage loop 测试先失败

**Step 3: 实现 Stage 3A 最小闭环**

这一步是整个计划的第一优先级, 也是最直接吸收 `Long-LRM` post-prediction optimization 的位置.

必须明确:

- `GaussianAdapter.freeze_for_stage("stage3a")`
- `GaussianAdapter.build_optimizer("stage3a", ...)`
- `StageController.should_stop_stage3a(...)`
- `compute_stage3a_losses(...)`

推荐最小 loss 组合:

- `L_rgb_weighted_native`
- `L_perceptual_patch` 的占位接口(允许先 stub)
- `L_opacity_sparse`
- `L_scale_ceiling`

这里先把“先 appearance, 后 geometry”的工程节奏固定死.

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_stage3a.py
```

Expected:

- Stage 3A 的冻结 / loss / loop contract 全部通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/runner.py src/refinement_v2/stage_controller.py src/refinement_v2/gaussian_adapter.py src/refinement_v2/losses.py tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_stage3a.py
git commit -s -m "feat: implement appearance first post refinement"
```

**Done when:**

- `Long-LRM` 风格的 post-prediction optimization 主线已经有了最小工程闭环
- Stage 3A 明确只做 native cleanup,不提前把 SR 混进来
- 默认流程可以只跑到 Stage 3A 就结束

---

### Task 6: 单独补上 opacity / pruning 方法论

**Files:**
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Modify: `src/refinement_v2/stage_controller.py`
- Create: `tests/refinement_v2/test_pruning.py`

**Step 1: 先写测试, 别让 pruning 只停留在口头**

至少覆盖:

- 低 opacity 高斯会被标成 prune candidate
- prune 前后统计会写入 diagnostics
- pruning 不会在 warmup 前触发
- 一次 pruning 不会删掉超过上限比例

建议锁住一个最小 contract:

```python
def test_pruning_respects_max_fraction() -> None:
    adapter = make_adapter_with_opacity([0.9, 0.01, 0.02, 0.03])
    kept = adapter.prune_low_opacity(
        threshold=0.05,
        max_fraction=0.25,
    )
    assert kept.num_gaussians == 3
```

**Step 2: 跑测试, 让 pruning 行为先红**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_pruning.py tests/refinement_v2/test_diagnostics.py
```

Expected:

- pruning 行为与 diagnostics 输出先失败

**Step 3: 实现最小 pruning controller**

这一步是第二优先级, 不能并入别的任务里含混带过.

至少实现:

- `GaussianAdapter.collect_prune_candidates(...)`
- `GaussianAdapter.prune_low_opacity(...)`
- `diagnostics.write_prune_summary(...)`
- `StageController.should_prune_now(iteration=...)`

要加的保护:

- warmup 迭代前不 prune
- 每次最多 prune 固定比例
- prune 后立刻重算 `opacity_lowconf_ratio`

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_pruning.py tests/refinement_v2/test_diagnostics.py
```

Expected:

- pruning 单测通过
- diagnostics 能记录 prune 前后统计

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/gaussian_adapter.py src/refinement_v2/diagnostics.py src/refinement_v2/stage_controller.py tests/refinement_v2/test_pruning.py tests/refinement_v2/test_diagnostics.py
git commit -s -m "feat: add opacity pruning controller"
```

**Done when:**

- opacity / pruning 已经从“方法论”变成可执行逻辑
- 这条能力可以和 Stage 3A 一起跑, 也可以单独开关

---

### Task 7: 接入 `SplatSuRe-style` selective SR patch supervision

**Files:**
- Modify: `src/refinement_v2/data_loader.py`
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/losses.py`
- Modify: `src/refinement_v2/weight_builder.py`
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Create: `tests/refinement_v2/test_patch_supervision.py`
- Read: `src/rendering/gs_deferred_patch.py`

**Step 1: 先写 patch 监督测试**

至少覆盖:

- patch window 采样不会越界
- SR patch 坐标能映射回 native / reference 尺度
- patch 渲染使用 `intrinsics_ref`
- `reference_mode=native` 时也能复用同一 patch path
- `gaussian_fidelity_score` 能转成每视图 `W_sr_select`
- `W_final_sr = W_robust * W_sr_select` 的 shape 与数值范围正确
- `L_sampling_smooth` 不会替换掉原有 `L_scale_ceiling`

**Step 2: 跑测试, 让 patch contract 先失败**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_patch_supervision.py
```

Expected:

- patch 采样或 intrinsics 对齐相关断言先失败

**Step 3: 实现最小 selective SR patch supervision**

在 `runner.py` 里加:

- `sample_patch_windows(...)`
- `render_patch_prediction(...)`
- `gather_reference_patch(...)`
- `compute_gaussian_fidelity_score(...)`
- `build_sr_selection_maps(...)`

在 `weight_builder.py` 里加:

- `build_sr_selection_weight(...)`
- `combine_sr_weights(w_robust, w_sr_select)`

在 `losses.py` 里让下面几项都支持 patch 张量:

- `L_rgb_weighted_sr`
- `L_perceptual_patch_sr`
- `L_sampling_smooth`

强约束:

- patch 渲染与全帧渲染共用同一套高斯参数解释
- SR 模式下不把 supervision 偷偷缩回 native 全帧
- `W_sr_select` 不是 `W_robust` 的替代
- `W_final_sr` 必须显式由两者组合而来

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_patch_supervision.py
```

Expected:

- SR patch / native patch 两类测试都通过
- fidelity / selection / final SR weight 三类测试都通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/data_loader.py src/refinement_v2/runner.py src/refinement_v2/losses.py src/refinement_v2/weight_builder.py src/refinement_v2/gaussian_adapter.py src/refinement_v2/diagnostics.py tests/refinement_v2/test_patch_supervision.py
git commit -s -m "feat: add selective sr patch supervision"
```

**Done when:**

- 超分后的对等视频终于能以“选择性高分辨率监督”的身份进入后置 refinement
- `gaussian_fidelity_score`、`W_sr_select`、`W_final_sr` 三个对象都已落地
- `L_sampling_smooth` 已作为 `Mip-inspired` 互补约束进入主线
- `sample.py` 主链依然不用改

---

### Task 8: 实现可选的 Stage 3B limited geometry

**Files:**
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/stage_controller.py`
- Modify: `src/refinement_v2/losses.py`
- Test: `tests/refinement_v2/test_stage_controller.py`
- Create: `tests/refinement_v2/test_runner_stage3b.py`

**Step 1: 先写失败测试**

至少覆盖:

- 未显式启用 `--enable-stage3b` 时不会进入 Stage 3B
- Stage 3B 允许更新:
  - scale
  - rotation
  - means
- `means` 更新被 `means_delta_cap` 限制
- `L_means_anchor` 和 `L_rotation_reg` 会加入总 loss

**Step 2: 跑测试, 让 Stage 3B contract 先红**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_stage_controller.py tests/refinement_v2/test_runner_stage3b.py
```

Expected:

- Stage 3B 开关与位移约束测试先失败

**Step 3: 实现 limited geometry**

这一步仍然保持克制:

- 只做小幅 geometry release
- 不做 pose
- 不做 densify-and-clone

至少实现:

- `GaussianAdapter.enable_stage3b_geometry(...)`
- `GaussianAdapter.clamp_means_delta(...)`
- `compute_stage3b_losses(...)`
- `StageController.should_enter_stage3b(...)`

进入条件要保守:

- Stage 3A 已完成
- selective SR 已完成
- ghosting 仍明显
- diagnostics 更像局部结构重叠, 不是整体 pose 错位或 SR 假细节污染

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_stage_controller.py tests/refinement_v2/test_runner_stage3b.py
```

Expected:

- Stage 3B 的 gating 与几何约束测试通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/gaussian_adapter.py src/refinement_v2/runner.py src/refinement_v2/stage_controller.py src/refinement_v2/losses.py tests/refinement_v2/test_stage_controller.py tests/refinement_v2/test_runner_stage3b.py
git commit -s -m "feat: add limited geometry refinement stage"
```

**Done when:**

- Stage 3B 只是可选后续阶段, 不会抢走 Stage 3A 的默认地位

---

### Task 9: 补齐 state / resume / 输出目录布局

**Files:**
- Modify: `src/refinement_v2/state_io.py`
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Create: `tests/refinement_v2/test_state_io.py`

**Step 1: 先写失败测试**

至少覆盖:

- `config_effective.yaml` 会落盘
- `scene_bundle.json` 会落盘
- `diagnostics.json` / `metrics.json` 会落盘
- `state/*.pt` 能保存和恢复当前 stage / iter / gaussian 参数
- `--resume` 时能找到最近一次 checkpoint

**Step 2: 跑测试, 让状态恢复路径先失败**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_state_io.py
```

Expected:

- state / resume / artifact layout 相关测试先失败

**Step 3: 实现状态与产物写入**

输出目录结构必须与 spec 对齐:

```text
outdir/
  config_effective.yaml
  scene_bundle.json
  gaussian_fidelity_histogram.json
  sr_selection_stats.json
  diagnostics.json
  metrics.json
  gaussians_refined.ply
  videos/
    baseline_render.mp4
    final_render.mp4
  residual_maps/
  weight_maps/
  sr_selection_maps/
  state/
```

至少实现:

- `save_checkpoint(...)`
- `load_latest_checkpoint(...)`
- `write_effective_config(...)`
- `write_metrics(...)`
- `write_diagnostics(...)`

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_state_io.py
```

Expected:

- resume 和输出布局测试全部通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/state_io.py src/refinement_v2/runner.py src/refinement_v2/diagnostics.py tests/refinement_v2/test_state_io.py
git commit -s -m "feat: add refinement state and resume outputs"
```

**Done when:**

- 后置 refinement 可以断点续跑
- dry-run / baseline / refined 三类产物边界清楚

---

### Task 10: 做最小集成验证与 dry-run 验收

**Files:**
- Modify: `scripts/refine_robust_v2.py`
- Modify: `src/refinement_v2/runner.py`
- Create: `tests/refinement_v2/test_smoke_dry_run.py`

**Step 1: 先写 smoke 测试**

至少覆盖:

- `--dry-run` 会执行:
  - CLI parse
  - SceneBundle build
  - baseline render
  - diagnostics write
- `--dry-run` 不会执行:
  - optimizer.step()
  - pruning
  - Stage 3B
- `--resume --dry-run` 组合不会崩溃

**Step 2: 跑测试, 先把 smoke contract 定死**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_smoke_dry_run.py
```

Expected:

- dry-run contract 先失败

**Step 3: 实现 dry-run 最终链路**

这里不要追求真实大场景全跑通.
先追求“流程是闭的”.

需要做到:

- `python3 scripts/refine_robust_v2.py ... --dry-run` 能稳定结束
- 输出至少包含:
  - `config_effective.yaml`
  - `scene_bundle.json`
  - `diagnostics.json`
  - `metrics.json`

**Step 4: 运行完整验证命令**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/gaussians_0.ply \
  --pose assets/demo/static/diffusion_output_generated_one/3/pose/demo.npz \
  --intrinsics assets/demo/static/diffusion_output_generated_one/3/intrinsics/demo.npz \
  --reference-video assets/demo/static/diffusion_output_generated_one/3/rgb/demo.mp4 \
  --reference-mode native \
  --scene-index 0 \
  --view-id 3 \
  --outdir outputs/refine_v2/dry_run_demo \
  --dry-run
```

Expected:

- `tests/refinement_v2` 全部通过
- dry-run 退出码 0
- 输出目录按 spec 生成最小产物

**Step 5: 提交本任务(可选)**

```bash
git add scripts/refine_robust_v2.py src/refinement_v2/runner.py tests/refinement_v2/test_smoke_dry_run.py
git commit -s -m "feat: add dry run validation for post refinement"
```

**Done when:**

- 新路线已经从“spec + 任务清单”推进到“可验证的最小工程闭环”

---

## Current Continuation Order (Use This As The Current Backlog Order)

截至 `2026-03-10` 的最新实验刷新, 这个 backlog 顺序已经再次更新。

原因是当前又多了六条关键新证据:

- `HR-only` 对照已经证明: 直接把 `lambda_lr_consistency` 设成 `0` 会显著伤害 native-space 指标, 不能把“删 LR”当成当前主线答案
- `Phase E` 的最小版 `stage3b` 已经真实跑通, 而且在最小 smoke 里相对同 run 的 `stage3sr` 已经出现正向收益
- `stage3b` 的独立超参数面也已经真正落地并完成真实 smoke, 因此接下来不需要再为 `Phase E` 补 CLI / config 基础设施
- 更长的 auto-gate run 已经证明: 不是每条长跑都会自动进入 `stage3b`, 因为 `stage3sr` 可能先把 residual 压到当前 gate 以下
- 但从同一条 `stage3sr` 末态继续接 `stage3b`, 已经再次拿到稳定正向收益, 而且 `start_stage=stage3b --resume` 的正式 CLI workflow 也已经验证通过
- 第一轮 continuation calibration 也已经证明: 只把 `iters_stage3b` 从 `32` 提到 `64`, 指标还能继续明显改善, 所以 `32 iter` 不能再当作充分预算

因此如果从今天的代码现实继续往前走, 推荐顺序应改成:

1. 继续推进 `Phase E`
   - 直接基于正式 `start_stage=stage3b --resume` workflow 继续做 `Phase E` 内部 calibration
   - 第一轮已回答:
     - `iters_stage3b=64` 明显优于 `iter32`
   - 当前更有信息量的下一刀变成:
     - 在 `iters_stage3b>=64` 的前提下量化 `means_delta_cap_stage3b`
     - 再量化 `lambda_means_anchor_stage3b`
     - 再量化 `lambda_rotation_reg_stage3b`
2. 在有一轮小范围 calibration 之后, 再决定 auto gate 是否要放宽
   - 因为现在已经能分清两件事:
     - auto gate 没放行
     - continuation 继续跑 `stage3b` 仍然有收益
   - 同时第一轮 calibration 还说明:
     - 预算本身就是变量
   - 所以下一步应先回答“默认要不要继续放 geometry”, 而不是继续猜 `Phase E` 本身是否成立
3. `Phase C` 的 `0.4 / 0.6` 近邻 sweep 暂时继续后移
   - `0.6` 的中间证据没有显示出明显优势
   - `0.0` 已经明确证明不能直接删 LR
   - 当前更有信息量的是沿着已经成立的 `Phase E` 做 geometry continuation calibration
4. `Phase B` 继续后移
   - 只有当 continuation 再次暴露 coverage 或显存问题时, 才回头重做“全图 weighted SR”

## Historical Execution Order

如果只是想追溯这份文档最初的 0-to-1 拆解顺序, 可以按下面理解:

1. 基础骨架:
   - Task 1
   - Task 2
   - Task 3
   - Task 4
   - Task 5
2. 第二优先级:
   - Task 6
3. 选择性高分辨率监督:
   - Task 7
4. 可选受限几何:
   - Task 8
5. 可靠性和交付:
   - Task 9
   - Task 10

## 明确不在本计划内

- `sample.py` 主链改造成高分辨率输入
- pose optimization
- joint pose + gaussian optimization
- Long-LRM 的长序列 token budget 控制
- `Mip-Splatting` 的 2D Mip filter(renderer-level 改造)
- `EDGS-style` local reinitialization
- `tttLRM` 的 fast-weight memory / LaCT / streaming reconstruction
- `depth anchor`
  - 截至 `2026-03-09`, 当前已弃用
  - 不作为这份计划的 continuation item

## 一句话继续顺序

先按新的 `Phase C` 口径把目标函数层级改对: `SR 主监督 + LR consistency`, 再补 HR 导出, 然后才视结果决定是否还需要回头重做全图 weighted SR 或进一步放开 geometry.

---

## 2026-03-06 历史代码现实对齐注记

这一节记录的是当时把文档从“方案讨论”切到“开始落代码”时的代码现实.
其中提到的很多动作现在已经完成, 所以它更适合作为历史上下文, 不应再被理解成当前未完成清单.

- 这份计划最初按“从 0 到 1”写任务.
- 但按当前代码现实,真正的第一刀不再是从空白实现 `Stage 3A`.
- 当前 `src/refinement_v2/runner.py` 已经有:
  - `run_stage2a()`
  - `run_stage2b()`
  - patch sampling / patch render / patch loss helper
- 问题在于:
  - `run_stage2a()` 把 native cleanup 和 patch supervision 混在了一起
  - 所以后续实现应先做一次“无行为变化拆边界”,再继续叠 selective SR

- 推荐先按下面的代码边界重排:
  1. `run_stage3a_native_cleanup()`
     - 对应当前 `run_stage2a()` 中不含 patch supervision 的 native loop
  2. `run_phase3s_build_sr_selection()`
     - 不做 optimizer
     - 只负责:
       - 拿 renderer `meta`
       - 计算 `gaussian_fidelity_score`
       - 生成 `W_sr_select`
  3. `run_stage3sr_selective_patch()`
     - 复用现有 patch 采样 / patch render 路径
     - 显式引入 `W_final_sr = W_robust * W_sr_select`
     - 在这里挂 `L_sampling_smooth`

- 还需要补一个关键现实:
  - `src/rendering/gs.py` 当前已经拿到 `rasterization(...).info`
  - 但没有往上返回
  - `Phase 3S` 的首个前置动作应是先把 `render_meta` 抬到 runner 层

- `gaussian_fidelity_score` 第一版也不建议上来就做 appearance fidelity.
- 当前 `.pixi` 环境里,`gsplat` 在 `packed=False` 时会返回 dense `meta`,其中稳定可用的字段至少包括:
  - `radii`
  - `means2d`
  - `opacities`
  - `tiles_per_gauss`
- 因此第一版口径建议先做:
  - `native support sufficiency`
  - 即:
    - `visible_mask`
    - `tiles_per_gauss` 归一化
    - `opacity_gate`
    - 再按 `max_view` 聚合成 per-Gaussian score

- `L_sampling_smooth` 的放置位置也在这里明确:
  - 放在 `src/refinement_v2/losses.py` 的 geometry regularizer 一侧
  - 不塞进 `_compute_patch_losses()`
  - Stage 3SR 默认启用
  - Stage 3B 可选延续

- 因此,如果现在开始实现,最顺手的顺序应改成:
  1. `src/rendering/gs.py` 返回 `render_meta`
  2. `src/refinement_v2/runner.py` 抽出 `Stage 3A / Phase 3S / Stage 3SR`
  3. `src/refinement_v2/weight_builder.py` 落第一版 `gaussian_fidelity_score / W_sr_select`
  4. `src/refinement_v2/losses.py` 再补 `L_sampling_smooth`

这些 1 ~ 4 项现在都已经落地.
