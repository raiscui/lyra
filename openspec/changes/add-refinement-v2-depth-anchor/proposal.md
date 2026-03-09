## Why

`refinement_v2` 当前在 `Stage 2A / Stage 3SR` 只依赖 RGB 残差做 appearance-first 优化, 没有延续训练期那类 depth 一致性约束. 这会让优化过程更容易把表面抹厚, 产生从正面看像点、从侧面看沿视线方向被拉长的空间歧义, 因而需要尽快补上一个与当前高斯坐标系天然对齐的 depth anchor.

## What Changes

- 为 `refinement_v2` 增加一个可配置的 depth anchoring 能力, 默认使用 baseline render depth 作为自锚点监督.
- 在 `Stage 2A / Stage 3SR` 中新增 depth consistency loss, 用来约束优化后渲染深度不要偏离 baseline 高斯的参考深度分布.
- 为 depth anchor 增加开关、权重和作用阶段配置, 允许先以最小侵入方式启用, 再逐步扩展到更强监督来源.
- 明确记录 V1 的非目标:
  - 不要求第一版依赖数据集 GT depth.
  - 不要求第一版依赖 `MoGe` / `ViPE` 外部深度.
  - 不要求第一版改变 `Stage 2B` 的几何更新策略.

## Capabilities

### New Capabilities
- `refinement-v2-depth-anchoring`: 为 `refinement_v2` 提供基于参考深度的渲染一致性约束, 先覆盖 `Stage 2A / Stage 3SR` 的 baseline render depth anchor, 以减少空间厚化和侧视拉丝.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `src/refinement_v2/losses.py`
  - `src/refinement_v2/runner.py`
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/data_loader.py` 或相关 scene bundle 契约
- Affected outputs:
  - refinement diagnostics
  - stage metrics
  - stage configuration surface
- Dependencies:
  - 复用现有深度归一化思路, 不新增外部模型依赖
