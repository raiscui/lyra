## 1. 配置与状态接线

- [ ] 1.1 在 `src/refinement_v2/config.py` 的 `StageHyperParams` 与参数映射层加入 depth anchor 配置项, 至少覆盖开关/权重/alpha 阈值/参考来源
- [ ] 1.2 在 `src/refinement_v2/runner.py` 中新增 baseline depth anchor 的缓存状态, 明确 reference depth、reference alpha、skip reason 的承载位置
- [ ] 1.3 在 appearance 流程开始前实现 baseline reference 捕获逻辑, 确保 `Stage 3SR` 复用同一份 `Stage 2A` 前参考

## 2. 深度损失与阶段接入

- [ ] 2.1 在 `src/refinement_v2/losses.py` 增加 depth normalization 与 `compute_depth_anchor_loss(...)`, 语义对齐训练期的 scale-invariant depth loss
- [ ] 2.2 在 `src/refinement_v2/runner.py` 的 `_run_appearance_stage(...)` 中接入 `loss_depth_anchor`, 只在 `Stage 2A / Stage 3SR` 且 reference 就绪时生效
- [ ] 2.3 保持 `Stage 2B` 与后续 geometry-moving 阶段不受 V1 depth anchor 影响, 避免把新逻辑误接到 `means` 更新路径

## 3. 诊断与兼容性

- [ ] 3.1 为 depth anchor 增加 diagnostics 输出, 包括 `loss_depth_anchor`、有效像素比例、reference source、skip reason
- [ ] 3.2 验证单卡与多卡 view-shard 渲染路径都能拿到并合并 `depths_pred`, 且 reference / prediction 设备对齐不出错
- [ ] 3.3 实现缺失 `depths_pred` 或有效 mask 为空时的 graceful fallback, 保证 refinement 继续运行

## 4. 测试与回归验证

- [ ] 4.1 为 `src/refinement_v2/losses.py` 新增单测, 覆盖 depth normalization、mask 过滤和 loss 数值稳定性
- [ ] 4.2 为 `src/refinement_v2/runner.py` 新增或补齐回归测试, 覆盖 reference 只捕获一次、`Stage 3SR` 复用 reference、`Stage 2B` 不启用 depth anchor
- [ ] 4.3 增加 graceful fallback 测试, 覆盖 `depths_pred` 缺失或空 mask 时仅记录 warning 而不崩溃
- [ ] 4.4 运行相关测试与最小 smoke refinement 命令, 确认 change 可进入实现阶段
