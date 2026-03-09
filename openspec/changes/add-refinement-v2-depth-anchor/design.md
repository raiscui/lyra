## Context

`refinement_v2` 当前已经具备三块与 depth anchor 直接相关的基础能力:

- renderer 已经返回 `depths_pred` 与 `alphas_pred`
- 多卡 view shard 聚合逻辑已经支持合并 `depths_pred`
- appearance-first 主循环已经统一收敛在 `_run_appearance_stage(...)`

但与此同时, 当前实现也有三个明确约束:

- dataloader 在 refinement 入口里显式关闭了 `use_depth`
- `losses.py` 只有 RGB / scale / opacity / patch / means / rotation 等损失, 没有 depth loss
- `Stage 2A / Stage 3SR` 是 appearance-first 阶段, 目标是先压制表面厚化和侧视拉丝, 而不是直接重写几何更新策略

因此这次设计的重点不是再引入一套新的外部几何来源, 而是把仓库里已经存在的渲染深度输出, 变成一个稳定、可控、可回退的 self-anchor.

## Goals / Non-Goals

**Goals:**

- 在 `Stage 2A / Stage 3SR` 中增加 depth consistency 约束, 抑制 RGB-only 优化导致的空间厚化
- 复用现有 renderer `depths_pred` 输出, 不为 V1 增加新的外部模型依赖
- 让 depth anchor 在单卡和多卡 view-shard 渲染路径下都能工作
- 保持实现边界清晰, 为后续接入 `dataset depth` 或 `MoGe/ViPE depth` 预留扩展位

**Non-Goals:**

- V1 不要求 dataloader 重新开启 GT depth 读取
- V1 不要求引入 `MoGe` / `ViPE` 外部深度作为强依赖
- V1 不改变 `Stage 2B` 的几何优化目标和参数冻结策略
- V1 不尝试重新定义 renderer 的 depth 物理含义, 仅消费现有输出契约

## Decisions

### 1. 参考深度来源选择 `baseline_render`

选择:

- V1 使用初始高斯在当前相机轨迹上的 baseline render depth 作为唯一参考深度来源

原因:

- 它与当前高斯处在同一坐标系, 不需要额外对齐
- 它不依赖 demo 资产是否带 depth 文件
- 它可以直接复用当前 renderer 输出, 接入成本最低

备选方案:

- `dataset GT depth`
  - 优点: 更接近真实几何
  - 缺点: 当前 refinement loader 明确关闭了 `use_depth`, 直接引入会扩大 I/O 和数据契约改动面
- `MoGe/ViPE external depth`
  - 优点: 后续可以作为纠偏项
  - 缺点: 需要补 provenance、尺度对齐和资产可用性策略, 不适合作为第一步强依赖

### 2. 参考深度在进入 `Stage 2A` 前捕获一次, 并跨 `Stage 3SR` 复用

选择:

- 使用当前初始高斯在 appearance 优化开始前渲染一次, 生成 immutable baseline reference
- `Stage 3SR` 复用同一份 baseline reference, 不在 `Stage 3SR` 开始时重新采样

原因:

- 这能确保 `Stage 3SR` 继续约束“不要偏离原始 baseline 几何”
- 如果在 `Stage 3SR` 前重新采样, 会把 `Stage 2A` 已经产生的深度漂移重新合法化

备选方案:

- 每个 stage 单独捕获 reference
  - 优点: 实现简单
  - 缺点: anchor 会逐阶段漂移, 削弱“防退化”作用

### 3. depth loss 采用训练期同类的尺度不变归一化语义

选择:

- 在 `src/refinement_v2/losses.py` 中新增 depth normalization / depth loss helper
- 语义与训练期 `normalize_depth(...) + smooth_l1_loss(...)` 保持一致
- valid mask 以参考深度有效像素为基础, 并叠加参考 alpha 阈值过滤空背景

原因:

- refinement 需要的是“保持相对深度结构”, 不是拟合绝对 metric depth
- 训练期已经证明这种 normalization 对不同尺度和偏移更稳
- 把 helper 留在 `refinement_v2/losses.py` 可以避免直接耦合训练脚本的其它 loss 依赖

备选方案:

- 直接 raw L1 / L2 depth
  - 优点: 实现最短
  - 缺点: 对全局 scale / bias 更敏感, 容易把 loss 变成数值尺度问题
- 直接 import `src/models/utils/loss.py`
  - 优点: 少写几行
  - 缺点: 会把 refinement 绑定到训练侧工具文件, 后续边界更乱

### 4. depth anchor 注入共享的 `_run_appearance_stage(...)`, 不直接散落到每个 stage

选择:

- 在 `_run_appearance_stage(...)` 中统一读取 `depths_pred`
- 在满足配置与 reference 就绪时追加 `loss_depth_anchor`
- 将其接入 `loss_total` 和 diagnostics metrics

原因:

- `Stage 2A` 与 `Stage 3SR` 共享同一主循环, 在这里接入最不容易分叉
- 多卡渲染路径已经在共享 loop 之前完成输出聚合, 不需要重复分支实现

备选方案:

- 分别在 `run_stage2a` / `run_stage3sr_selective_patch` 单独加 loss
  - 优点: 看起来更直观
  - 缺点: 容易出现两个 stage 逻辑漂移

### 5. V1 只约束 `Stage 2A / Stage 3SR`, 明确不进入 `Stage 2B`

选择:

- V1 depth anchor 只进入 appearance-first 阶段
- `Stage 2B` 继续沿用现有 `means_anchor + rotation_reg + patch supervision` 体系

原因:

- 当前用户最关心的是“前面 refinement 把表面抹厚”
- `Stage 2B` 一旦接 depth, 就会从“防退化锚点”升级成“几何纠偏策略”, 风险面明显更大

备选方案:

- 直接把 depth loss 同时接入 `Stage 2B`
  - 优点: 理论上更能限制几何漂移
  - 缺点: 会和 `means` 更新、patch loss、rotation regularization 形成更复杂的权重耦合

### 6. 缺失 depth reference 时优雅降级, 不阻断现有 refinement 流程

选择:

- 当 renderer 未返回 `depths_pred`, 或 reference 构造失败时:
  - 记录 warning / skip reason
  - depth anchor 自动失活
  - 其余 refinement 流程继续执行

原因:

- 当前 change 是增强现有流程, 不是让旧资产在缺少 depth 契约时直接不可运行

备选方案:

- 直接 hard fail
  - 优点: 问题更显眼
  - 缺点: 会把一个增强项变成兼容性破坏

## Risks / Trade-offs

- [参考深度来自 baseline render, 不是 GT] → 只能保证“防退化”, 不能保证“纠正初始化错误”; 后续若要纠偏, 再扩展 external depth source
- [额外 depth loss 可能压制某些必要的 appearance 调整] → 默认把权重单独配置, 并先只作用于 `Stage 2A / Stage 3SR`
- [alpha mask 阈值过高会丢掉细结构, 过低会把背景噪声纳入监督] → 提供阈值配置并在 diagnostics 里输出有效像素占比
- [多卡路径需要 reference 与 pred depth 设备对齐] → 统一在共享 loop 中做 device 对齐, 避免单独 stage 分支复制逻辑

## Migration Plan

1. 在 `StageHyperParams` 与 CLI/config 映射层加入 depth anchor 配置项
2. 在 `RefinementRunner` 中引入 reference capture / cache / diagnostics 状态
3. 在 `losses.py` 增加独立的 depth normalization 与 depth anchor loss helper
4. 把 `loss_depth_anchor` 接入 `_run_appearance_stage(...)`
5. 为单卡和多卡路径补回归测试, 覆盖:
   - reference 构造
   - loss 计算
   - stage scope
   - graceful fallback
6. 默认以保守权重开启或在配置中显式开启, 首轮只验证 `Stage 2A / Stage 3SR`

回滚策略:

- 将 `lambda_depth_anchor` 设回 `0`
- 或直接关闭 depth anchor 开关
- 不需要迁移数据文件

## Open Questions

- `depths_pred` 的背景值在所有 renderer 模式下是否都稳定满足 `<= 0` 或 `alpha≈0` 的过滤假设
- V1 默认值是否直接启用 depth anchor, 还是先默认关闭、由 demo 配置显式打开
- 后续如果引入 `dataset depth` / `MoGe/ViPE depth`, 是否要统一为 `depth_anchor_source = baseline_render | dataset | external`
