# Joint Refinement Camera Gaussians V2 Implementation Plan

这份计划已经完成,并已在 2026-03-07 归档。

## 归档位置

- 完整归档副本:
  - `docs/plans/archive/2026-03-06-joint-refinement-camera-gaussians-v2.md`

## 当前该看什么

- 如果你要看这份 v2 主线的最终完成记录、历史任务清单、以及归档时的最新状态:
  - 请看 `docs/plans/archive/2026-03-06-joint-refinement-camera-gaussians-v2.md`
- 如果你要继续看当前仍在推进的增强主线:
  - 请看 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`

## 归档说明

- 原始 `Task 1 ~ Task 13` 已全部完成
- 归档前已重新追平到最新状态:
  - `tests/refinement_v2` 当前验证为 `82 passed`
  - direct file inputs(`--pose-path` / `--intrinsics-path` / `--rgb-path`) 已不再是未来项
  - full-view root mode(`--scene-stem` / `--view-ids` / `--pose-root` / `--intrinsics-root` / `--rgb-root` / `--reference-root`) 已落地
  - 当前正式推荐主线为 `native baseline -> Stage 2B`
