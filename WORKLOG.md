# WORKLOG

## [2026-03-10 00:20:00 UTC] 任务名称: 分析 Stage 3SR patch supervision 数据流与可扩展挂点

### 任务内容
- 读取 `src/refinement_v2/runner.py` 与 `weight_builder.py`、`data_loader.py`、`losses.py`、相关测试
- 静态梳理 `Stage 3SR` 的 patch supervision 数据流
- 判断现有 reference 分辨率整图渲染 / 多 patch 渲染入口是否已存在
- 找出升级到多 patch / 更大 coverage weighted SR supervision 的最小挂点

### 完成过程
- 先确认上游 reference supervision 由 `SceneBundle.reference_images` / `intrinsics_ref` 承载
- 再梳理 `run_stage2a -> run_phase3s_build_sr_selection -> run_stage3sr_selective_patch` 的阶段边界
- 继续下钻 `_compute_patch_losses()` 到 `sample_patch_windows()` / `render_patch_prediction()` / `combine_sr_weights()`
- 同时核对多卡分支, 确认 `_compute_patch_losses()` 也是多卡 shard 路径的统一消费点
- 参考 `tests/refinement_v2/test_patch_supervision.py` 验证 patch path 确实支持 native/SR 共用与 reference intrinsics offset

### 总结感悟
- 当前代码里真正值得复用的不是“一个完整的多 patch 系统”, 而是三块已经到位的基础设施:
  - reference 监督数据
  - reference 分辨率的 `sr_selection_map`
  - `_compute_patch_losses()` 这个统一接线点
- 单热点 patch 只是当前 supervision 策略层的选择, 不是整个 Stage 3SR 架构不可改的硬约束

## [2026-03-10 00:31:00 UTC] 任务名称: 更正 Stage 3SR 多 patch 现状判断

### 任务内容
- 回读 `runner.py` 后半段, 修正上一条分析里关于多 patch 入口的错误判断

### 完成过程
- 发现 `src/refinement_v2/runner.py:857` 已有 `sample_sr_patch_window_sets(...)`
- 发现 `src/refinement_v2/runner.py:1060` 已在 `_compute_patch_losses(...)` 内循环多个 patch set 并求平均 loss
- 发现 `src/refinement_v2/config.py:77` / `src/refinement_v2/config.py:313` 已暴露 `sr_patches_per_view`
- 发现 `tests/refinement_v2/test_patch_supervision.py:229` / `tests/refinement_v2/test_patch_supervision.py:294` 已覆盖该路径

### 总结感悟
- 这条主线已经不再是“有没有多 patch”问题, 而是“多 patch 的 priority / coverage 是否还要继续扩大”问题

## [2026-03-10 03:34:14 UTC] 任务名称: 盘点当前机器上可清理的磁盘垃圾候选

### 任务内容
- 只做只读排查, 不执行任何删除操作。
- 识别系统缓存、临时目录、项目环境、实验输出、模型权重这几类磁盘占用。
- 给出可清理性分类, 方便后续按优先级清理。

### 完成过程
- 先按项目规范回读并续档六文件, 为本轮任务开新批次上下文。
- 用 `df`、`findmnt`、`du`、`find`、`lsof +L1` 逐层排查根盘、`/root`、`/workspace`、`/tmp` 与重点缓存目录。
- 识别出几个主要占用来源:
  - `~/.cache/huggingface/hub` 约 `90G`
  - `~/.cache/pip` 约 `7.4G`
  - `/tmp` 下若干临时目录, 最大约 `3.7G`
  - 三个仓库的 `.pixi` 环境约 `11G ~ 14G`
  - `lyra` 的 `checkpoints` 约 `31G`
  - `lyra` 的 `outputs` 约 `3.5G`
- 按“相对安全清理”“可重建但成本高”“不建议先动”三类完成了整理。

### 总结感悟
- 在 overlay 环境里, 顶层 `du -x /` 很容易误导, 必须直接打到热点路径。
- 真正的清理顺序不该只看目录大小, 还要先区分缓存、环境、实验证据、模型本体这几类性质。

## [2026-03-10 03:41:59 UTC] 任务名称: 清理 `/tmp` 历史临时文件

### 任务内容
- 按用户要求, 只清理 `/tmp`。
- 在删除前先检查 `/tmp` 是否存在仍被进程占用的活跃条目, 避免误删当前会话依赖的 socket 或临时文件。

### 完成过程
- 先用 `lsof -nP +D /tmp` 检查正在占用 `/tmp` 的文件与目录。
- 本轮未发现需要保留的活跃顶层条目。
- 随后删除 `/tmp` 下未被占用的历史临时目录、日志和调试残留。
- 清理后再次复核:
  - `/tmp` 约 `44K`
  - 根盘可用空间约 `61G -> 62G`

### 关键结果
- 实际释放约 `5.48G`。
- 主要被清掉的大目录包括:
  - `/tmp/longsplat-no-source`
  - `/tmp/tttLRM`
  - `/tmp/gradio`
  - `/tmp/LongSplat`
  - `/tmp/mermaid-validate`

### 总结感悟
- `/tmp` 这类目录适合先做一轮“活跃占用检查”, 再清历史残留。
- 这样既能保证安全, 也能最大化回收空间。

## [2026-03-10 03:48:00 UTC] 任务名称: 落地 `Closest-to-SplatSuRe Track Phase B` 第一轮多 patch supervision

### 任务内容
- 把 `Stage 3SR` 从单 patch 路径推进到多 patch supervision
- 补 CLI、测试、runner 逻辑
- 用 full-view `sub8` 真实验证 `multi4` / `multi2`

### 完成过程
- 在 `src/refinement_v2/config.py` 新增 `--sr-patches-per-view`
- 在 `src/refinement_v2/runner.py` 新增:
  - `_build_sr_patch_priority_map()`
  - `sample_sr_patch_window_sets()`
  - `_compute_patch_losses()` 的多 patch 平均逻辑
- 在 `tests/refinement_v2/test_config.py` 与 `tests/refinement_v2/test_patch_supervision.py` 补了对应测试
- 跑完相关回归后, 做了两轮真实验证:
  - `multi4` 在 `stage3sr` OOM
  - `multi2` 完整跑通, 但最终指标只略好于旧 baseline, 同时略弱于 `Phase A`

### 总结感悟
- 这轮说明“从单 patch 到多 patch”本身不是难点, 真正的难点在于把更大的 supervision coverage 转成有效收益
- 当前最小版 `Phase B` 更像是把基础设施搭起来了, 但还没把收益曲线拉出来

## [2026-03-10 06:10:00 UTC] 任务名称: fidelity 参数接线与 full-view `sub8` calibration

### 任务内容
- 把 `WeightBuilder` 内部已有但未暴露的 fidelity 超参数打通到 CLI / config / runner
- 补回归测试, 防止后续再次出现“参数表面存在, 实际一直吃默认值”的回归
- 在 full-view external SR + `sub8` 口径下做一轮最小 calibration

### 完成过程
- 在 `src/refinement_v2/config.py` 中新增 fidelity 超参数字段与 CLI 参数:
  - `fidelity_ratio_threshold`
  - `fidelity_sigmoid_k`
  - `fidelity_min_views`
  - `fidelity_opacity_threshold`
- 在 `src/refinement_v2/runner.py` 初始化 `WeightBuilder(...)` 时补齐上述参数透传
- 在 `tests/refinement_v2/test_config.py` 与 `tests/refinement_v2/test_runner_stage2a.py` 补测试
- 跑定向回归:
  - `58 passed`
- 随后跑 full-view `sub8` 真实实验:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_thr1p1_sub8_iter20_20260310`
  - 只改 `--fidelity-ratio-threshold 1.1`

### 总结感悟
- 这轮把“fidelity 参数其实没有真正生效”的隐性缺口补上了。
- 但动态证据也说明, 即便把 selection coverage 明显放大, 最终指标仍没有超过当前 `Phase A`。
- 因此后续更值得继续的主线, 已经从“再调 routing 强度”逐步转向“换目标设计”, 也就是 `Phase C`。

## [2026-03-10 06:35:00 UTC] 任务名称: 按新共识调整 `Phase C` 方案与任务

### 任务内容
- 根据用户刚确认的新方向, 重写 `Closest-to-SplatSuRe Track` 里 `Phase C` 的方案定义
- 同步调整推荐执行顺序与当前 backlog 顺序
- 把新的实施任务拆清楚, 作为后续真正写代码前的正式入口

### 完成过程
- 回读 `Phase C`、`Recommended Execution Order` 与 `Current Continuation Order`
- 把旧口径从“native 主损失 + SR 辅助”改成:
  - SR 6 视频主监督
  - native LR downsample consistency
- 在文档里补出新的 `Phase C` 当前实施任务:
  - full-frame HR render
  - HR -> LR consistency
  - `Stage 3SR` 目标重组
  - diagnostics / tests
- 同时把 backlog 顺序更新为:
  - `Phase C -> Phase D -> 必要时再回头看 Phase B -> Phase E`

### 总结感悟
- 这一步很重要, 因为它把后续实现从“继续在旧目标上微调”纠正成了“先把目标函数本身改对”。
- 现在后续代码任务已经不再含糊, 可以直接进入 `Phase C` 的正式实现。


## [2026-03-10 06:35:00 UTC] 任务名称: 完成 `Phase C` 的 full-frame HR 路径并跑通真实 `sub8` smoke

### 任务内容
- 修复 `Stage 3SR` 的 `Phase C` 半完成状态
- 补齐 `full-frame HR supervision + LR consistency` 的回归测试
- 在真实 full-view external SR + `sub8` 数据上验证 `Phase C` 是否能跑通

### 完成过程
- 先修复 `_get_reference_images(...)` 的 `device/dtype` 接口缺口, 让 full-frame HR 路径不再因 helper 签名不匹配直接报错
- 更新 `test_config.py`、`test_losses.py`、`test_runner_stage2a.py`、`test_patch_supervision.py`, 把 Phase C 的 config、downsample 和 full-frame HR 口径补齐
- 跑定向回归, 确认 `61 passed`
- 第一轮真实 smoke 发现 OOM 落在 `gsplat.rasterization()`
- 随后把 native render 改成“depth anchor 关闭时不参与反传”, 再跑一轮真实 smoke
- 第二轮 OOM 被推迟到 `combine_sr_weights()`, 证据表明单纯 serial render 仍然会在后续 full concat 时把峰值重新拉回来
- 最终新增 `_iter_scene_single_device_view_shards()` 并把 `Phase C` 改成真正的 stream-sharded loss/backward
- 第三轮真实 smoke 成功落地:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_streamshard_20260310`

### 总结感悟
- “把 HR render 切 shard”不等于真的降低峰值.
- 如果后面仍然把所有 shard 拼回整块 tensor, 峰值迟早还会回来.
- 对 `Phase C` 这种 full-frame supervision, 正确的显存落地方式是:
  - shard 级前向
  - shard 级 loss
  - shard 级 backward
  - metrics 再 detach 到 CPU 汇总


## [2026-03-10 07:05:00 UTC] 任务名称: 完成 `Phase C` 的 lambda sweep 与 iter8 真实对照

### 任务内容
- 围绕已经跑通的 `Phase C` 路径做小范围 `lambda_hr_rgb` sweep
- 再把选中的配置拉到更长 iter 的真实 `sub8` 对照
- 验证 `Phase C` 不只是能跑, 而且在更长阶段里是否开始出现稳定收益

### 完成过程
- 先读取最小 smoke 的 loss 构成, 确认 `loss_hr_rgb / loss_lr_consistency ≈ 0.0275`
- 据此没有继续试 `1/2` 这种过小权重, 而是直接做:
  - `hr=8, lr=1`
  - `hr=16, lr=1`
  - `hr=32, lr=1`
- 三组 `1 iter` 真实 `sub8` sweep 都完整跑通, 但几乎没有指标差异
- 随后把 `hr=32, lr=1` 拉到 `iter8`:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr1_sub8_iter8_20260310`
- 该 run 最终成功生成 `diagnostics.json`、`metrics_stage3sr.json`、`gaussians_stage3sr.ply` 与 `final_render.mp4`

### 总结感悟
- `Phase C` 当前已经从“能不能跑”进入了“长程收益曲线怎么长出来”的阶段。
- `1 iter` smoke 适合排错, 但已经不适合继续区分 loss 配比优劣。
- 现在真正值得继续扫的, 已经从 `lambda_hr_rgb` 单独抬高, 逐步转向:
  - 更长 iter
  - 或开始下调 `lambda_lr_consistency`

## [2026-03-14 00:20:00 UTC] 任务名称: 定位 `diffusion_output_generated_my` 生成链路与相机资产语义

### 任务内容
- 只读排查 `/workspace/lyra/assets/demo/static/diffusion_output_generated_my`
- 定位真正写出 `rgb/*.mp4`、`pose/*.npz`、`intrinsics/*.npz` 的脚本与函数
- 验证 `pose`、`intrinsics` 与 `0..5` 子目录的语义, 并区分事实与推断

### 完成过程
- 先从 `README.md` 精确命中生成命令, 确认目标目录由 `gen3c_single_image_sdg.py --video_save_folder .../diffusion_output_generated_my` 生成
- 再用 `rg` 和 `ast-grep` 抓出 `np.savez(...)`、`save_video(...)`、`generate_camera_trajectory(...)` 的落盘位置
- 回读 `gen3c_single_image_sdg.py` 中 normal path 与 resume path 两套写盘分支
- 继续回读 `camera_utils.py`、`radym.py`、`provider.py`、`refinement_v2/data_loader.py` 与测试, 核对下游对 `pose/intrinsics` 的解释契约
- 最后直接读取真实 `dj-style.npz` 与 `dj-style.mp4` 元数据, 用动态证据确认帧数、分辨率、四元组与轨迹方向

### 总结感悟
- 这类目录的核心生成器不是 `sample.py`, 而是 static SDG 脚本 `gen3c_single_image_sdg.py`
- `0..5` 目录名本质上是六条固定轨迹的 `traj_idx`, 后续才被当作 multi-view 的 `view_id`
- `pose.npz` 当前资产语义不是 renderer 内部的 `cam_view`, 而是 raw `c2w`; 只有在 refinement 入口处才被转换成 `inverse(c2w).T`

## [2026-03-10 07:47:33 UTC] 任务名称: 回收 Phase C iter20 长跑结果并完成文档收口

### 任务内容
- 回收 `hr32 + lr0.5 + iter20 + sub8` 的真实运行结果
- 将其与 `Phase C iter8` 和 `Phase A iter20` 做 apples-to-apples 对比
- 同步更新 `docs/cmd.md` 与长期计划文档, 明确当前 backlog 顺序

### 完成过程
- 先确认长跑目录已落盘并检查核心产物是否完整
- 读取 `metrics_stage3sr.json` 的最终 step 与 `diagnostics.json` 的 artifacts
- 形成三组对照结论后, 更新:
  - `docs/cmd.md`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
  - 六文件中的 `task_plan.md` / `notes.md`

### 总结感悟
- `Phase C` 这条线现在已经不只是“概念上更对”, 而是拿到了真实长程收益证据。
- 当 native 指标 gap 已经缩到较小量级时, 下一阶段的主瓶颈往往会从“目标函数是否成立”转移到“交付产物是不是用户真正想看的输出”。

## [2026-03-10 08:11:59 UTC] 任务名称: 完成 Phase D 最小 HR 导出闭环

### 任务内容
- 在 `Phase C` 已有 reference-space 训练能力的基础上, 补齐 `Phase D` 的最终 HR 导出
- 让 diagnostics 明确拆出 native-space 与 hr-space 摘要
- 用测试回归证明这次改动没有破坏既有路径

### 完成过程
- 在 `src/refinement_v2/runner.py` 中复用现有 `reference-space` scene builder, 补出 evaluation/export 版 HR render helper
- 导出层新增:
  - `baseline_render_hr`
  - `gt_reference_hr`
  - `final_render_hr`
- `diagnostics.json` 现在额外包含:
  - `native_hw`
  - `reference_hw`
  - `baseline_hr`
  - `final_hr`
- 在 `tests/refinement_v2/helpers.py` 扩展测试夹具, 允许直接构造 `super_resolved` synthetic scene
- 新增并通过 HR 导出相关测试, 之后又跑通 `tests/refinement_v2` 全套 `113 passed`

### 总结感悟
- 当训练层已经有 `reference-space` helper 时, 导出层最好的做法通常不是“再开一套新流程”, 而是把 evaluation/export 逻辑顺着既有 scene builder 接出来。
- 这类改动最容易被忽略的不是训练 loss, 而是最终 diagnostics 是否真的把多空间指标拆清楚。

## [2026-03-10 08:13:02 UTC] 任务名称: 完成 Phase D 的 HR 导出并做真实 smoke 验证

### 任务内容
- 在不破坏现有 native 导出的前提下, 补出 `Phase D` 的 HR 导出链路
- 为 `diagnostics.json` 补齐 `baseline_hr / final_hr` 与 HR 增益字段
- 用定向回归、全量回归和真实 full-view smoke 三层证据验证结果

### 完成过程
- 在 `src/refinement_v2/runner.py` 中补了:
  - `_reference_space_enabled()`
  - `_render_reference_scene_for_evaluation()`
  - `_summarize_reference_prediction()`
- 扩展 baseline / final 导出, 让 `baseline_render_hr`、`gt_reference_hr`、`final_render_hr` 自动落盘
- 扩展最终 diagnostics summary, 增加:
  - `native_hw`
  - `reference_hw`
  - `baseline_hr`
  - `final_hr`
  - `psnr_gain_hr`
  - `sharpness_gain_hr`
  - `residual_mean_hr_drop`
- 更新测试夹具和回归测试后, 跑通:
  - `29 passed`
  - `113 passed`
- 最后补了一条真实 full-view external SR `phase0` smoke:
  - `outputs/refine_v2/phaseD_phase0_hr_export_smoke_20260310`

### 总结感悟
- `Phase D` 这种任务, 真正关键的不是“再导一份视频”, 而是把 native-space 和 reference-space 的交付语义分清楚。
- 一旦 final artifact 也进入 HR 空间, 前面 `Phase C` 的价值才真正变成可见产物, 而不是只停留在训练指标里。

## [2026-03-10 08:53:00 UTC] 任务名称: 收口 Phase C iter32 结果并重排当前 backlog

### 任务内容
- 回读 `Phase C hr32 lr0.5 sub8 iter32` 的真实实验产物
- 把最新证据同步到六文件与长期文档
- 按照新证据调整当前 continuation backlog

### 完成过程
- 重新读取三组关键 run 的 `diagnostics.json`, 确认 `Phase C iter32`、`Phase C iter20` 与 `Phase A iter20` 的最终对照关系
- 读取 `metrics_stage3sr.json` 最后一个点, 确认本轮仍然是 `full_frame_hr` 监督口径
- 因 `task_plan.md` 已超过 1000 行, 先续档到 `archive/task_plan_20260310_084544.md`, 再新建当前任务计划
- 同步更新:
  - `docs/cmd.md`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
- 把已回读的历史六文件移入 `archive/`, 降低根目录噪音

### 总结感悟
- 当主线实验已经跨过 baseline 的关键门槛时, backlog 必须跟着换挡, 不能继续围着旧问题打转。
- 这轮最重要的不是又多出一个目录, 而是把问题重新定义成“剩下这点 residual gap 怎么办”。

## [2026-03-10 09:49:30 UTC] 任务名称: 实现 Phase E 最小版 `stage3b`

### 任务内容
- 停止继续 `Phase C` 参数调优
- 在现有 `Phase C` / `Stage 3SR` 主监督路径上实现 `Phase E`
- 让 SR 信息第一次能通过独立阶段去推动有限 geometry 更新

### 完成过程
- 先回读 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md` 的 `Phase E` 设计和现有 `stage2b` 实现
- 再把 `stage3sr` 的 full-frame HR reference-space 主监督循环抽成共享 helper
- 在不重写第二套 geometry pipeline 的前提下, 新增:
  - `enable_stage3b`
  - `should_enter_stage3b(...)`
  - `run_stage3b()`
  - `compute_stage3b_losses(...)`
  - `tests/refinement_v2/test_runner_stage3b.py`
- 跑通了:
  - 定向测试 `25 passed`
  - 全量回归 `117 passed`
  - 真实 smoke `outputs/refine_v2/phaseE_stage3b_smoke_sub8_20260310`

### 总结感悟
- `Phase E` 最稳的切口, 不是复制一份 `stage2b`, 而是复用 `Phase C` 已经跑通的 HR 主监督路径, 再只把 geometry 这一层有限放开。
- 这样做的好处是, 新阶段的“监督语义”是连续的, 不会刚进入 geometry 就突然切回另一套目标函数。

## [2026-03-10 10:20:00 UTC] 任务名称: 为 Phase E 补齐 `stage3b` 独立超参数面

### 任务内容
- 把 `stage3b` 从 `stage2b` 的共享 geometry 配置中拆出来
- 为 `Phase E` 增加独立的 iteration / regularizer / means clamp 参数入口
- 用测试和真实 smoke 验证这些参数已经真实进入运行时

### 完成过程
- 在 `src/refinement_v2/config.py` 中新增 `stage3b` 专属字段和 CLI, 同时用兼容 fallback 保持已有命令不变
- 在 `src/refinement_v2/runner.py` 中新增 geometry 参数解析 helper, 让 `stage3b` 单独读取自己的 budget 和 regularizer
- 在 `src/refinement_v2/gaussian_adapter.py` 中让 `stage3b` 的 clamp 优先使用 `means_delta_cap_stage3b`
- 新增和更新测试:
  - `tests/refinement_v2/test_config.py`
  - `tests/refinement_v2/test_gaussian_adapter.py`
  - `tests/refinement_v2/test_runner_stage3b.py`
- 跑通了:
  - 定向回归 `30 passed`
  - 全量 `tests/refinement_v2` `119 passed`
- 最后补了一条真实 smoke:
  - `outputs/refine_v2/phaseE_stage3b_hparams_smoke_sub8_20260310`

### 总结感悟
- 新阶段如果长期借用旧阶段的超参数, 很容易让后续实验归因变脏。
- 最稳的做法不是立刻大调参, 而是先把“独立参数面”这层基础设施做干净, 再去跑长程对照。

## [2026-03-10 11:22:00 UTC] 任务名称: 收口 Phase E 长程结果并固化 `start_stage=stage3b`

### 任务内容
- 跑更长的 `Phase E stage3b` apples-to-apples 对照
- 分析为什么 auto-gate 长跑没有进入 `stage3b`
- 把 continuation workflow 固化成正式 CLI 入口

### 完成过程
- 先跑了更长 auto-gate run:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_hr32_lr0p5_sub8_iter32_20260310`
- 再读取 `diagnostics.json` 与 `state/latest.pt`, 确认它停在 `stage3sr` 的原因是:
  - `residual_mean` 已低于 `0.045` 的 `local_overlap_persistent` 门槛
  - 也就是 gate 没放行, 不是 `stage3b` 自己失败
- 随后做了最小 continuation 实验:
  - 手工从 `stage3sr` 末态续跑 `stage3b`
  - 结果继续优于纯 `stage3sr`
- 再把这个路径正式工程化:
  - `config.py` 支持 `start_stage=stage3b`
  - `runner.py` 新增 `bootstrap_stage3b_from_current_gaussians()` 与 `warm_start_stage3b`
  - `test_runner_stage3b.py` 新增 start-stage 回归
- 最后用正式 CLI 再跑一条真实 continuation:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
  - 并验证它与手工 continuation 最终结果几乎一致

### 总结感悟
- auto gate 回答的是“默认还要不要继续放 geometry”, 不是“继续放 geometry 会不会还有收益”。
- 当这两件事开始分离时, 最正确的做法不是立刻改 gate, 而是先补一个正式 continuation 入口, 把问题拆开验证。

## [2026-03-10 11:45:00 UTC] 任务名称: 完成 Phase E 第一轮 continuation calibration

### 任务内容
- 基于正式 `start_stage=stage3b --resume` workflow 做第一轮 `stage3b` 校准
- 不先乱动 regularizer, 先单独验证 `iters_stage3b` 是否不足
- 统一保持 `--target-subsample 8`

### 完成过程
- 先读取 `iter32` continuation 的 `metrics_stage3b.json`, 发现尾部连续改善, 因而把“预算可能偏短”设为主假设
- 复制 source run 的 `state/latest.pt`, 启动真实 continuation:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter64_20260310`
- 只把 `iters_stage3b` 提到 `64`, 其余 `stage3b` 参数保持不变
- 跑完后读取 `diagnostics.json` 与 `metrics_stage3b.json`, 完成对 `iter32` continuation 和 `stage3sr` source run 的双重对比

### 总结感悟
- 这轮最重要的不是“64 比 32 好一点”, 而是正式把一个模糊判断变成了证据:
  - `32 iter` 明显不够
- 因为这一刀只改了预算, 所以后续如果再扫 cap / regularizer, 归因会干净很多

## [2026-03-14 16:26:00 UTC] 任务名称: 收敛 sdg 推理输出文件名, 避免 prompt 直拼路径

### 任务内容
- 修复 `gen3c_single_image_sdg.py` 与 `gen3c_dynamic_sdg.py` 生成过长输出文件名的问题
- 让默认命名不再包含空格和标点
- 保留对历史长文件名产物的 resume 兼容

### 完成过程
- 先用 `rg` 和 `ast-grep` 定位 `clip_name` 的生成链路, 确认问题点在两个 `*_sdg.py` 的 `_build_clip_name(...)`
- 对照 `gen3c_single_image.py` 与 `add_common_arguments()` 的 `video_save_name` 语义, 决定不新增 CLI, 而是改良现有命名策略
- 新建轻量模块 `cosmos_predict1/diffusion/inference/output_naming.py`, 集中放置:
  - 安全文件名清洗
  - `--video_save_name` 优先逻辑
  - 默认 `输入 stem + 短 prompt hash`
  - legacy 长文件名回退构造
- 在 `gen3c_single_image_sdg.py` 与 `gen3c_dynamic_sdg.py` 中新增 `_resolve_output_plan(...)`, 让新命名优先, 但历史目录里已有旧产物时仍可继续 resume
- 补充 `tests/test_inference_output_naming.py`, 再跑语法检查和定向测试完成验证

### 总结感悟
- 这次最值得复用的规律不是“怎么截断 prompt”, 而是“命名规则要同时考虑新产物可读性和旧产物续跑兼容性”
- 另外, 纯字符串 helper 不应塞进重推理模块里。否则连最小单测都会被运行时依赖绑架

## [2026-03-14 16:58:00 UTC] 任务名称: 统一 google-t5 默认加载目录到固定本地路径

### 任务内容
- 修复 `google-t5/t5-11b` 默认仍可能回落到 `~/.cache` 的问题
- 让 `CosmosT5TextEncoder` 自身默认值与主 pipeline 使用同一份固定目录
- 补最小回归测试, 锁住默认行为

### 完成过程
- 先用 `rg` 和 `ast-grep` 定位 T5 的加载链路, 确认:
  - `BaseWorldGenerationPipeline` 已显式传 `/model/HuggingFace/google-t5/t5-11b`
  - 但 `cosmos_predict1/auxiliary/t5_text_encoder.py` 的默认 `cache_dir` 仍是 `~/.cache`
- 随后把 `DEFAULT_T5_MODEL_NAME` 与 `DEFAULT_T5_MODEL_DIR` 收敛到 `t5_text_encoder.py`
- 再让 `BaseWorldGenerationPipeline` 直接复用这两个常量, 消除“双份字符串”
- 最后新增 `tests/test_t5_text_encoder.py`, 分别验证:
  - 直接 `CosmosT5TextEncoder()` 默认走固定本地目录
  - 主 pipeline 传参也走同一份常量

### 总结感悟
- 这次真正需要修的不是“主入口有没有传对”, 而是“类默认值和主入口是否一致”
- 只要底层默认值没收口, 新入口一出现, 问题就会回潮
