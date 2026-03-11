# EPIPHANY_LOG

## [2026-03-10 03:34:14 UTC] 主题: overlay 容器里不要只信顶层 `du -x /`, 要直接打到热点路径

### 发现来源
- 本轮做整机磁盘垃圾排查时, `df` 显示根盘已使用约 `242G`。
- 但首次运行 `du -xhd1 /` 只看到了几百 MB 的可见目录, 与真实情况严重不符。

### 核心问题
- 在当前 overlay / snapshotter 环境里, 顶层 `du -x /` 容易被层叠文件系统行为误导。
- 如果因此直接得出“容器里几乎没东西可清”, 结论会明显失真。

### 为什么重要
- 后续只要再遇到磁盘排查、缓存清理、环境瘦身这类任务, 都可能先被这一步带偏。
- 更稳的做法是:
  - 先用 `df` 看总量
  - 再直接点查 `~/.cache`、`/tmp`、工作区 `.pixi`、`outputs`、`checkpoints` 等热点路径
  - 必要时再用 `lsof +L1` 排除“已删除但仍占用”的文件

### 未来风险
- 如果继续按顶层 `du -x /` 判断, 很容易错过真正的大头目录。
- 如果把多个单独 `du` 结果机械相加, 又可能被共享硬链接反向误导成“总量更大”。

### 当前结论
- overlay 环境中的磁盘排查, 需要用“热点目录逐个核对”的策略。
- 单目录容量适合做清理优先级判断, 但跨目录求和要保留硬链接重复计数的警惕。

### 后续讨论入口
- 下次如果用户要继续清理, 先从:
  - `~/.cache/huggingface/hub`
  - `~/.cache/pip`
  - `/tmp`
  - 各仓库 `.pixi`
  开始做逐项确认, 不要再先跑一轮顶层 `du -x /` 就收工。

## [2026-03-10 03:48:00 UTC] 主题: `Phase B` 的多 patch 基础设施已经足够, 但收益瓶颈已经转移到 supervision 强度和目标设计

### 发现来源
- 本轮完成了 `Phase B` 的最小多 patch 实现, 并跑了 full-view `sub8` 的 `multi4` / `multi2` 两轮真实实验

### 核心问题
- `multi4` 直接 OOM, 说明简单堆 patch 数量会先撞上算力/显存边界
- `multi2` 虽然完整跑通, 但最终指标只比旧 baseline 略好, 同时略弱于 `Phase A`
- 这说明当前问题已经不再是“有没有多 patch”, 而是:
  - 多 patch 之后的监督强度如何分配
  - 以及当前 `Stage 3SR` 目标是否足够承接更大的 coverage

### 为什么重要
- 这会直接改变下一步优先级
- 如果继续只堆 patch 数量, 很容易先 OOM, 但收益仍然不明显
- 更值得做的要么是:
  - fidelity / selection 强度参数校准
  - 要么直接进入更像论文的 `Phase C`

### 当前结论
- `Phase B` 的基础设施 blocker 已解除
- 但“multi-patch = 明显增益”这个命题目前没有被证据支持

### 后续讨论入口
- 下一轮优先考虑:
  1. 暴露 fidelity / selection 强度参数做小范围 calibration
  2. 直接推进 `Phase C` 的 `HR render + LR consistency`

## [2026-03-10 06:10:00 UTC] 主题: Phase A 的瓶颈已经不再只是 fidelity routing 强度

### 发现来源
- 本轮先补了 fidelity 参数 CLI 接线, 再做了一轮 full-view external SR + `sub8` 的最小 calibration
- calibration 只改一项:
  - `fidelity_ratio_threshold: 1.5 -> 1.1`

### 核心问题
- 这轮 calibration 明显放大了 selection coverage:
  - `selection_mean: 0.01574 -> 0.04699`
- 但最终 `PSNR / residual_mean / sharpness` 仍没有超过当前 `Phase A` rerun

### 为什么重要
- 这说明当前问题已经不再是“selection 太保守所以没收益”这么简单
- 即便把 routing 强度明显推高, 收益也没有同步放大
- 继续只围着 threshold / k / min_views 打转, 很可能会继续得到“coverage 变化明显, final metrics 变化很小”的结果

### 当前结论
- `Phase A` 现在更像已经完成了“routing 语义校正”
- 但要把 external SR 真的转成更显著收益, 更关键的下一步应是:
  - `Phase C: HR render + LR consistency`
- 换句话说, 当前主瓶颈更像 supervision objective, 而不是 selection routing 本身

### 后续讨论入口
- 下次如果继续 Closest-to-SplatSuRe Track, 优先从:
  - `src/refinement_v2/runner.py` 的 `_compute_patch_losses()` / HR render 分支
  - `Stage 3SR` 目标改成 `HR output + LR downsample consistency`
  开始, 不要再默认先堆更多 routing calibration

## [2026-03-10 06:35:00 UTC] 主题: 当前真正需要改的已经不是 routing, 而是 supervision hierarchy

### 发现来源
- 先前的 fidelity calibration 证明, 仅靠放大 selection coverage 并没有把最终指标继续拉高
- 随后的代码核对又确认, 当前实现仍然是:
  - native GT 主损失
  - SR reference 辅助监督
- 用户随后明确接受新的方向:
  - SR 6 视频升格为主监督
  - native LR 退到 consistency

### 核心问题
- 如果继续沿旧层级推进, 后面做再多 Phase A / Phase B 微调, 都可能只是在旧目标上打转
- 真正该先修正的是“谁是主监督, 谁是兜底一致性”这个 hierarchy

### 为什么重要
- 这会直接决定 `Phase C` 的接口设计、loss 结构、metrics 拆法和显存策略
- 一旦 hierarchy 不改, 后面很多实现细节都会建立在错误目标上

### 当前结论
- 从今天开始, `Closest-to-SplatSuRe Track` 的主线口径应切到:
  - SR 主监督
  - LR consistency
- 这不是对旧实现的小修小补, 而是目标函数层级的重新对齐

### 后续讨论入口
- 继续实现时优先从:
  - full-frame HR render helper
  - HR -> LR downsample consistency helper
  开始, 再谈 selective weighting 和导出层


## [2026-03-10 06:35:00 UTC] 主题: "serial render" 不是显存优化的终点, 如果后面还会 full concat, 峰值迟早回来

### 发现来源
- `Phase C` 的 full-frame HR 路径第一次 OOM 在 renderer
- 把 native render 改成 `no_grad()` 后, 第二次 OOM 又转移到 `combine_sr_weights()`
- 随后改成真正的 stream-sharded loss/backward 后, 真实 `sub8` smoke 才完整跑通

### 核心问题
- 很多时候“我已经做了 shard render”会给人一种错误安全感
- 但如果后续还要把 shard 重新拼成完整大 tensor 再算 loss, 那只是在推迟峰值出现的位置

### 为什么重要
- 这条规律不只适用于当前的 `Phase C`
- 以后只要是:
  - full-frame supervision
  - 高分辨率 reference
  - 多视角 view batch
- 都要警惕这种“前向分片了, loss 还是整块”的假优化

### 当前结论
- 真正有效的显存优化要落到整个链路:
  - shard 级 render
  - shard 级 weight build
  - shard 级 loss
  - shard 级 backward
  - CPU 侧 metrics 汇总
- 只做前向分片, 不足以保证 full-frame HR 训练能落地

### 后续讨论入口
- 以后如果还要继续推进 `Phase D` 或更长 `Phase C` 训练, 先复用这套 `stream-sharded` 思路
- 不要回到“先拼完整 HR tensor, 再统一算 loss”的写法


## [2026-03-10 07:05:00 UTC] 主题: 当最小 smoke 已经可跑时, 继续堆更多 1-iter sweep 很可能只是在制造假工作量

### 发现来源
- 本轮 `Phase C` 的 `lambda_hr_rgb=8/16/32` 三组真实 `sub8` smoke 都已跑通
- 但三组在 `1 iter` 下几乎没有指标差异
- 同时 `iter8` 长跑却已经能看到明显变化

### 核心问题
- 很多时候“我又多跑了几个点”会给人一种在认真探索的错觉
- 但如果这些点仍停留在 `1 iter`, 它们可能根本没有能力区分配置优劣

### 为什么重要
- 这会直接影响后续实验预算分配
- 如果继续在无分辨率的 `1 iter` 区间里堆 sweep, 只会消耗 GPU 时间, 却不提供更强结论

### 当前结论
- 当前 `Phase C` 已经过了“最小 smoke 排错期”
- 后续参数实验要么:
  - 拉长 iter
  - 要么换更有信息量的维度(例如开始动 `lambda_lr_consistency`)
- 继续只堆更多 `1 iter` 的 `lambda_hr_rgb` 点, 价值很低

### 后续讨论入口
- 下次如果继续 `Phase C`, 优先从:
  - `lambda_lr_consistency`
  - 或更长 iter
  进入, 不要默认再扩 `hr=8/16/32/64/...` 的 1-iter sweep

## [2026-03-10 07:47:33 UTC] 主题: 当 Phase C 已经把 native gap 压到较小量级后, 下一个主瓶颈会从 objective validity 转移到 deliverable scale

### 发现来源
- 本轮完成了 `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter20_20260310` 的真实长跑回收
- 并与:
  - `Phase C hr32 lr1 iter8`
  - `Phase A iter20`
  做了直接对比

### 核心问题
- `Phase C` 现在已经不再停留在“能不能跑通”这个层面
- 它已经把 native 指标 gap 压到接近 `Phase A iter20`
- 但最终交付物仍主要是 native-space 导出, 这会遮蔽 `HR supervision` 真正带来的收益

### 为什么重要
- 如果继续只盯着 native `PSNR / residual_mean`, 很容易误判当前主线是否已经值得继续
- 当目标函数层级已经改对, 下一个最真实的产品缺口往往是:
  - 你最终能不能把 HR 输出直接交付出来
  - 而不是永远只在 LR 指标上比较

### 当前结论
- `Phase C` 的有效性已经被动态证据支持, 但尚未在 native 指标上完全超过 `Phase A`
- 因此当前推荐 backlog 应优先切到 `Phase D`
- 若未来还要继续压 `Phase C`, 也应围绕更长 iter 或 `lr≈0.5` 的进一步 sweep, 而不是回到 `1 iter` 点状扫描

### 后续讨论入口
- 下次继续时优先看:
  - `src/refinement_v2/diagnostics.py`
  - `src/refinement_v2/runner.py`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md` 中的 `Phase D`

## [2026-03-10 08:11:59 UTC] 主题: 当训练层已经有 reference-space scene builder 时, Phase D 最好的实现不是再起一套 HR 流程, 而是把 export plane 顺着既有 helper 接出来

### 发现来源
- 本轮推进 `Phase D` 时回读了 `runner.py` 里已有的:
  - `_build_reference_render_scene()`
  - `_render_reference_scene_for_training()`
  - `Phase C` 的 `psnr_hr / residual_mean_hr`
- 随后用最小代码把 HR 导出补到 baseline/final 路径, 并跑通了 `113 passed` 回归

### 核心问题
- 很容易把“我要支持 HR 输出”误解成“我要再写第二套 HR pipeline”
- 但这往往会把 scene 构造、renderer cache、diagnostics 命名再次分叉, 很快变成两套并行系统

### 为什么重要
- 当前项目已经在训练层证明了 `reference-space` 的可行性
- 所以下一个正确动作应该是把这条语义延伸到 export layer
- 而不是把同样的分辨率语义重新发明一遍

### 当前结论
- `Phase D` 的正确最小切口是:
  - 复用现有 `reference-space` helper
  - 单独补一条 evaluation/export 版 HR render
  - 然后在 diagnostics 里显式拆出 native/hr 摘要
- 这比新起一套 HR runner 更稳, 也更不容易产生长期分叉

### 后续讨论入口
- 下次如果继续扩展 `Phase D`, 优先看:
  - `src/refinement_v2/runner.py`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md` 的 `Phase D`
- 不要默认去新建第二套 HR-only 导出主流程

## [2026-03-10 08:13:02 UTC] 主题: 当 HR 导出真正成为最终产物后, Phase C 的价值才从“训练内部信号”变成“用户可见信号”

### 发现来源
- 本轮完成了 `Phase D` 的 runner 实现、回归验证和真实 full-view `phase0` smoke
- 真实目录为:
  - `outputs/refine_v2/phaseD_phase0_hr_export_smoke_20260310`

### 核心问题
- 之前即便 `Phase C` 已经能给出 `psnr_hr` 等训练指标, 最终交付仍主要是 native after 视频
- 这会导致一个认知偏差:
  - 训练内部已经开始朝 HR 目标优化
  - 但用户最后看到的仍主要是 LR 空间结果

### 为什么重要
- 这类问题本质上不是 objective 是否正确, 而是 deliverable scale 是否对齐
- 如果交付尺度和训练尺度错开, 再好的 HR supervision 也会被“最终看不到”所抵消

### 当前结论
- `Phase D` 已经证明, 一旦把 `baseline_render_hr / final_render_hr` 与 `baseline_hr / final_hr` 一起落盘
- `Phase C` 的收益就不再只是 optimizer 内部的数字, 而会变成真正可观察的 before / after 产物

### 后续讨论入口
- 下次如果继续主线优化, 优先把注意力放回:
  - `Phase C` 在 HR / native 双空间下的长期收益曲线
  - 而不是再回到“导出看不到 HR”这类交付层缺口

## [2026-03-10 08:53:00 UTC] 主题: 当主线已经跨过 baseline 的关键门槛后, backlog 必须从“成立性证明”切到“frontier 压缩”

### 发现来源
- 本轮重新读取了:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter32_20260310/diagnostics.json`
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_sub8_iter20_20260310/diagnostics.json`
- 并同步更新了 `docs/cmd.md` 与长期计划文档

### 核心问题
- 很多 backlog 会在主线已经拿到关键胜利后, 还停留在旧问题上。
- 现在的真实状态已经不是:
  - `Phase C` 能不能成立
  - `Phase D` 有没有 HR 导出
- 而是:
  - native `residual_mean` 这最后一点差距, 到底还能不能只靠当前 appearance objective 压过去

### 为什么重要
- 如果 backlog 不换挡, GPU 预算就会继续浪费在已经回答过的问题上。
- 这会直接推迟真正值得验证的 frontier:
  - `lambda_lr_consistency≈0.5` 的近邻 trade-off
  - 或 `Phase E` 的有限 geometry 自由度

### 当前结论
- `Phase C hr32 lr0.5 sub8 iter32` 已经在 native `psnr` 上超过 `Phase A iter20`。
- `Phase D` 也已经完成真实 HR 导出验证。
- 当前最合理的主线问题, 是 native `residual_mean` 能否也一起超过 `Phase A`, 而不是回到旧 objective 继续打转。

### 后续讨论入口
- 下一轮优先看:
  - `docs/cmd.md` 中的 `6E`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md` 的 `Current Continuation Order`
  - 新一轮 `lambda_lr_consistency≈0.5` 近邻 sweep 的真实结果

## [2026-03-10 09:49:30 UTC] 主题: 最小可用的 Phase E 不需要新起第二套 geometry pipeline, 关键是让 geometry release 继续吃同一套 HR 主监督

### 发现来源
- 本轮为实现 `Phase E`, 回读了现有 `stage3sr` full-frame HR 路径和 `stage2b` limited geometry 路径
- 随后完成了 `stage3b` 的最小实现与真实 smoke

### 核心问题
- 如果 `Phase E` 直接复制一份 `stage2b`, 很容易出现一个隐性问题:
  - 上一阶段还在吃 HR 主监督 + LR consistency
  - 一进入 geometry release 就突然换成另一套更偏 native 的 loss
- 这会让“SR 是否真正推动结构”这个问题本身变得不纯

### 为什么重要
- `Phase E` 最核心的价值不是“终于能动 means 了”
- 而是“geometry release 仍然沿着同一套 SR 主监督语义继续前进”
- 只有这样, 才能更真实地回答 SR 是否真的在改结构, 而不是只在改纹理

### 当前结论
- 最小可用的 `Phase E` 可以这样落:
  - 复用 `Phase C` 的 full-frame HR reference-space 主监督循环
  - 在此基础上只新增 geometry release 与对应 regularizers
  - 不必先复制第二套完整 geometry pipeline
- 真实 smoke 已经证明这条实现链路能跑到 `stage3b`, 而且在最小 smoke 中相对 `stage3sr` 已有正向收益

### 后续讨论入口
- 下一轮优先看:
  - `outputs/refine_v2/phaseE_stage3b_smoke_sub8_20260310`
  - 更长 iter 的 `stage3b` 对照
  - 是否需要给 `stage3b` 拆独立超参数面

## [2026-03-10 10:20:00 UTC] 主题: 新阶段如果继续复用旧阶段超参数, 会把后续实验归因直接污染掉

### 发现来源
- 本轮继续做 `Phase E` 时, 发现最小版 `stage3b` 虽然已经能跑, 但仍复用 `stage2b` 的 iteration / regularizer / means clamp
- 随后补了独立参数面, 并用真实 smoke 验证这些新参数已经进入运行时

### 核心问题
- 一个新阶段即便已经有独立代码路径, 只要仍绑定旧阶段超参数, 后续长跑就很难回答“收益到底来自新阶段语义, 还是来自旧阶段配置”

### 为什么重要
- 这类问题不会立刻让代码报错
- 但会在实验解释层面悄悄制造歧义, 让 GPU 预算被浪费在归因不清的对照上

### 当前结论
- `Phase E` 的正确继续方式不是立刻乱扫参数
- 而是先把 `stage3b` 的独立参数面补齐, 再做 apples-to-apples 长跑
- 真实 smoke 已证明:
  - `iters_stage3b`
  - `lambda_means_anchor_stage3b`
  - `lambda_rotation_reg_stage3b`
  - `means_delta_cap_stage3b`
  都已经真实进入运行时

### 后续讨论入口
- 下次继续时优先看:
  - `outputs/refine_v2/phaseE_stage3b_hparams_smoke_sub8_20260310`
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md` 的 `Phase E`
  - `docs/cmd.md` 的 `6G`

## [2026-03-10 11:22:00 UTC] 主题: auto gate 的“默认是否继续”与 continuation 的“继续后是否有收益”是两件不同的问题

### 发现来源
- 更长的 `Phase E` auto-gate run 停在了 `stage3sr`
- 随后从同一条 `stage3sr` 末态继续接 `stage3b`, 又拿到了明确正向收益
- 最后把这条 continuation 路径固化成了正式 CLI workflow

### 核心问题
- 很容易把“auto gate 没放行”误写成“这个阶段没有价值”。
- 但这两者不是一回事。

### 为什么重要
- 如果不区分这两层语义, 后续就会在错误的问题上浪费 GPU:
  - 明明该先问“默认门槛是否过保守”
  - 却又回到“Phase E 到底成立不成立”

### 当前结论
- 已验证:
  - auto gate 停在 `stage3sr`
  - continuation 继续跑 `stage3b` 仍然能带来 `+0.479 psnr` / `-0.002337 residual_mean` 等增益
- 所以下一步的正确问题应该是:
  - `stage3b` 内部超参数怎么调
  - auto gate 默认阈值要不要跟着放宽
- 而不是继续怀疑 `Phase E` 本身是否成立

### 后续讨论入口
- `outputs/refine_v2/full_view_sr_stage3b_phaseE_hr32_lr0p5_sub8_iter32_20260310`
- `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
- `docs/cmd.md` 的 `6H` / `6I`

## [2026-03-10 11:45:00 UTC] 主题: `Phase E` 现在的第一瓶颈已不再是“是否成立”, 而是 continuation 预算线设得太短

### 发现来源
- 本轮对正式 CLI continuation 做了第一轮 calibration
- 只把 `iters_stage3b` 从 `32` 提到 `64`, 其余 `stage3b` regularizer 与 cap 全部保持不变

### 核心问题
- 如果 `stage3b` 在 `iter32` 尾部还持续改善, 但 backlog 还把它当成“先扫 regularizer”, 那么后续很多 sweep 都会混入一个更基本的偏差:
  - 预算本身就没给够

### 为什么重要
- 这会让后续实验解释变脏
- 你会误以为某个 cap / regularizer 很关键, 但其实它只是在补偿一个过短的优化窗口

### 当前结论
- 已验证:
  - `iters_stage3b=64` 相比 `iter32` continuation 继续提升:
    - `psnr +0.329351`
    - `residual_mean -0.001469`
    - `psnr_hr +0.190583`
    - `residual_mean_hr -0.001184`
- 仍未验证:
  - 64 之后是否还应该继续加预算
  - 还是 `means_delta_cap_stage3b` / `lambda_*_stage3b` 开始成为新的主瓶颈

### 后续讨论入口
- `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter64_20260310`
- `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
- `docs/cmd.md` 的下一条 `Phase E calibration` 记录
