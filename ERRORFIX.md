# ERRORFIX

## [2026-03-10 03:30:00 UTC] 问题: 续档六文件时 Markdown 中的反引号导致 shell 写入内容被截断

### 现象
- 续档命令已经成功完成了文件重命名和归档。
- 但新生成的 `task_plan.md`、`notes.md` 中, 含反引号的段落出现了内容丢失。
- 终端同时出现了 `command not found`、`Is a directory` 等与 Markdown 文本本身无关的报错。

### 根因
- 外层命令使用了 `bash -lc '... '` 这一类单引号整体包裹脚本的写法。
- 脚本内部的 Markdown 正文又包含了单引号和反引号。
- 结果导致 shell 提前结束字符串, 后续文本被当作命令片段执行。

### 修复
- 不再用单引号整体包裹长段 Markdown 写入脚本。
- 改用 `python3` 直接重写六文件内容, 避免 shell 对正文做额外解释。
- 重新检查新文件内容, 确认反引号与路径均完整保留。

### 验证
- 重新查看 `task_plan.md` 与 `notes.md`, 反引号路径和文件名已经完整存在。
- 新文件不再出现被截断的 `archive/`、`/root`、`/workspace` 等字段。

## [2026-03-10 06:10:00 UTC] 问题: fidelity 超参数写在 `WeightBuilder` 里, 但 CLI 和 runner 实际没有接通

### 现象
- `WeightBuilder` 已有 `fidelity_ratio_threshold`、`fidelity_sigmoid_k`、`fidelity_min_views`、`fidelity_opacity_threshold`。
- 但之前从 CLI 无法设置这些值。
- 即使外部以为自己在做 `Phase A` calibration, 实际运行时仍然一直吃默认值。

### 根因
- `src/refinement_v2/config.py` 的 `StageHyperParams` 与 CLI parser 没有对应字段。
- `src/refinement_v2/runner.py` 初始化 `WeightBuilder(...)` 时也没有透传这些参数。
- 导致 fidelity 参数虽然在类定义里存在, 但在主流程里始终是隐形常量。

### 修复
- 在 `StageHyperParams` 中补齐 4 个 fidelity 字段。
- 在 CLI 中新增对应参数。
- 在 `RefinementRunner` 初始化 `WeightBuilder(...)` 时完整透传。
- 补测试覆盖:
  - CLI 默认值与显式映射
  - runner 初始化后的真实注入值

### 验证
- 定向回归:
  - `58 passed`
- 真实 calibration:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_thr1p1_sub8_iter20_20260310`
- 动态证据证明新参数已经真实生效:
  - 只改 `fidelity_ratio_threshold: 1.5 -> 1.1`
  - `selection_mean` 从 `0.01574` 提升到 `0.04699`

## [2026-03-10 06:36:00 UTC] 问题: `bash -lc` 的提示文本里直接包含反引号, 触发命令替换报 `command not found`

### 现象
- 一条单纯用于向用户报备的 `printf` 命令输出了:
  - `bash: line 1: Phase: command not found`
- 业务逻辑没有受影响, 但终端确实出现了错误输出。

### 根因
- 提示文本里直接包含了反引号包裹的 `` `Phase C` ``。
- 外层又走了 `bash -lc printf


## [2026-03-10 06:37:00 UTC] 更正: 上一条关于 shell 反引号的 ERRORFIX 记录被 shell 再次截断, 现补完整版本

### 现象
- 上一条 ERRORFIX 追加时, 因为正文里再次出现反引号, shell heredoc 写入又被截断。
- `ERRORFIX.md` 里留下了一条不完整记录, 结尾停在 `bash -lc printf`。

### 根因
- 我虽然意识到问题来自反引号命令替换, 但这次追加 ERRORFIX 时仍然用了 shell heredoc。
- 正文继续包含反引号示例, 导致同类问题再次发生。

### 修复
- 这次改用 `python3` 直接追加文本, 不再让 shell 解释正文。
- 正确结论如下:
  - `bash -lc` 的提示文本里如果直接包含反引号包裹的 `Phase C`, shell 会把其中内容当成命令替换。
  - 后续 shell 提示文本不再直接放这类反引号片段。

### 验证
- 本次使用 `python3` 追加后, 文件内容已完整落盘。
- 后续如需记录含反引号的长文本, 应优先继续使用 `python3` 或真正安全的单引号 heredoc。


## [2026-03-10 06:35:00 UTC] 问题: `Phase C` full-frame HR 路径在真实 `sub8` 上连续两层 OOM

### 现象
- 第一轮真实 smoke 会在 `gsplat.rasterization()` OOM。
- 修正 native render 的 autograd 后, 第二轮 OOM 被推迟到 `WeightBuilder.combine_sr_weights()`。
- 这说明之前的第一假设只解释了第一层峰值, 不能解释全部问题。

### 根因
- 第一层显存峰值来自:
  - full-frame HR 路径里 native render 其实并不参与主损失反传
  - 但实现仍保留了它的 autograd 图
- 第二层显存峰值来自:
  - `_render_scene_serial_view_shards()` 只是串行渲染
  - 后续又把所有 HR shard 重新 `torch.cat(...)` 成整块 tensor
  - full-frame 权重图和 LR consistency 也继续按全量张量一次性构造
- 所以旧实现本质上是“串行前向 + 整块 loss”, 不是真正的流式执行。

### 修复
- 先在 `depth anchor` 关闭时, 让 native render 走 `torch.no_grad()`。
- 再新增 `_iter_scene_single_device_view_shards()`。
- 把 `_run_stage3sr_full_frame_hr()` 重构成:
  - native residual / weight 先准备
  - HR render 按 shard 逐块前向
  - `L_hr_rgb` / `L_lr_consistency` 按 shard 逐块 backward
  - 指标只在 detach 到 CPU 后再汇总

### 验证
- 编译验证:
  - `python3 -m py_compile src/refinement_v2/config.py src/refinement_v2/losses.py src/refinement_v2/runner.py tests/refinement_v2/test_config.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_stage2a.py tests/refinement_v2/test_patch_supervision.py tests/refinement_v2/test_runner_phase0.py tests/refinement_v2/test_runner_stage2b.py tests/refinement_v2/test_depth_anchor.py`
- 回归验证:
  - `61 passed`
- 真实验证:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_streamshard_20260310`
  - `phase_reached = stage3sr`
  - `stage3sr_supervision_mode = full_frame_hr`
  - `metrics_stage3sr.json` / `diagnostics.json` / `gaussians_stage3sr.ply` / `final_render.mp4` 均已生成

## [2026-03-10 07:48:08 UTC] 问题: 追加 Markdown 时未加引号 heredoc 导致反引号内容被 shell 吞掉

### 问题
- 在向 `task_plan.md` 追加 Markdown 时, 使用了未加引号的 heredoc.
- 文本里包含反引号包裹的 `session_id=46472`, shell 把它当成 command substitution 处理, 结果正文被吞掉了一部分。

### 原因
- shell 中 `cat <<EOF` 会对正文做参数展开和命令替换.
- 只要正文里有反引号, 就有机会误触发执行或丢字.

### 修复
- 后续凡是正文里可能出现反引号的 Markdown 追加, 统一改成:
  - `cat <<'EOF'`
- 如果还需要插入时间戳等变量, 用外层 `printf` 打标题, heredoc 正文保持单引号保护.

### 验证
- 本次修正后已重新追加说明记录, 保留了原意并没有再触发反引号替换.
- 当前可见证据:
  - `task_plan.md` 末尾新增了本条修正说明
  - `ERRORFIX.md` 记录了原因与处理方式

## [2026-03-10 08:11:59 UTC] 问题: `refinement_v2` 测试命令需要同时满足 `PYTHONPATH` 与 `pixi` 环境

### 问题
- 直接跑 `pytest -q tests/refinement_v2` 时, 会因为基础环境缺少 `omegaconf` 等依赖而失败。
- 直接跑 `pixi run pytest -q tests/refinement_v2` 时, 又会因为没有 `PYTHONPATH` 而报 `ModuleNotFoundError: src`。

### 原因
- 仓库代码按 `src/...` 绝对导入组织, 因此测试运行时必须把仓库根目录放进 `PYTHONPATH`。
- 同时依赖又装在 `pixi` 环境里, 不能只用系统 Python。

### 修复
- `refinement_v2` 当前稳定的测试命令应统一写成:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2`

### 验证
- 使用上述命令后, 本轮回归结果为:
  - `113 passed`

## [2026-03-10 08:13:02 UTC] 问题: Python / pixi 两种测试口径都会因为环境不全而给出假失败

### 问题
- 直接运行:
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  会在部分全量测试上报:
  - `ModuleNotFoundError: No module named omegaconf`
- 直接运行:
  - `pixi run pytest -q tests/refinement_v2`
  又会在 collection 阶段报:
  - `ModuleNotFoundError: No module named src`

### 原因
- 前者用了宿主 Python, 缺少项目依赖环境。
- 后者虽然进了 pixi env, 但没有把仓库根目录注入 `PYTHONPATH`.

### 修复
- 当前仓库里跑全量 refinement 回归的正确口径应固定为:
  - `PYTHONPATH="$(pwd)" pixi run pytest -q tests/refinement_v2`

### 验证
- 本轮按上面口径重跑后, 实际结果为:
  - `113 passed`
- 这说明前两种报错都属于环境口径错误, 不是本次代码回归.
