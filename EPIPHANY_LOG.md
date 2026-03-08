## 2026-03-08 06:50 UTC 主题: 双卡不是充分条件, renderer meta 粒度才是决定因素

### 发现来源

- 在双 `RTX 4090` 上重试 `full-view native sub4` 时
- 第一轮多卡改造已经把 OOM 从 runner 主卡聚合点挪走
- 第二轮真实 trace 又在 `src/rendering/gs.py::_merge_chunk_meta(...)` 看到新的 OOM

### 核心问题

- “程序能看到两张卡” 不等于 “full-view 大 observation 就一定能跑”
- 如果 shard 粒度仍然过粗
- renderer 内部自己的 dense meta 也会把单卡打满

### 为什么重要

- 这条规律不只影响当前 `refinement_v2`
- 以后任何依赖 dense per-view render meta 的多卡优化路径
- 都不能只停留在“按设备数平均切 view”

### 未来风险

- 如果后续继续做:
  - 更高 observation 密度
  - `Stage 2B`
  - `Phase 4`
  - 或更接近 SplatSuRe 的全图 HR 监督
- 只要 renderer 仍返回 dense meta
- 就可能再次遇到“多卡已开启,但单 shard 仍 OOM”的二次显存墙

### 当前结论

- 当前已验证有效的做法是:
  1. 无 backward 阶段统一 CPU gather
  2. CUDA 多卡 shard 收成单 `view` 粒度
  3. patch render 不再套用全局多卡聚合路径
- 尚未验证的部分:
  - `Stage 2B / Phase 4` 的双卡训练路径
  - 更长正式预算下是否还会在别的 renderer meta 字段再触顶

### 后续讨论入口

- 下次如果继续推进双卡 full-view 长跑:
  - 先看 `outputs/refine_v2/full_view_native_stage2a_dual_gpu_sub4_smoke_20260308_0640`
  - 再看 `src/refinement_v2/runner.py` 的:
    - `_build_render_shards(...)`
    - `_run_appearance_stage_multi_device(...)`

## 2026-03-08 08:48 UTC 主题: 双卡长跑已接近既有最优盆地, 继续堆同类预算未必再有质变

### 发现来源

- 在正式长跑:
  - `outputs/refine_v2/full_view_native_stage2a_dual_gpu_sub4_baseline_v1_20260308_0718`
  完成后
- 对照:
  - `outputs/refine_v2/full_view_native_stage2a_fair_v3_20260308_0630`
  的最终 `diagnostics.json`

### 核心问题

- 更大显存与真双卡, 解决了“能不能跑”的问题
- 但没有自动带来“全指标明显跨档更好”
- 当前看到的是:
  - `PSNR` 略高
  - `residual_mean / sharpness` 并没有同步拉开

### 为什么重要

- 这意味着接下来的瓶颈, 很可能已经不是“算力不够所以跑不动”
- 而是“当前 objective / stage 设计本身只能收敛到这个水平附近”

### 未来风险

- 如果后面继续只做:
  - 同类参数
  - 同类阶段
  - 只是再加预算
- 很可能只会得到更贵的近似同水平结果

### 当前结论

- 当前已验证事实:
  - 双卡正式 baseline 最终:
    - `psnr = 24.54589251176849`
    - `residual_mean = 0.029650945216417313`
    - `sharpness = 0.003145787864923477`
  - `fair_v3` 最终:
    - `psnr = 24.52869587364757`
    - `residual_mean = 0.029334016144275665`
    - `sharpness = 0.0034818428102880716`
  - 两者最终 `num_gaussians` 都是:
    - `1137966`
- 目前最合理的判断是:
  - 这条 native full-view 主线已经接近同一优化盆地
  - 如果想继续拉开质量差距, 更可能需要:
    - `Stage 2B`
    - loss 重新配比
    - 或更接近 SplatSuRe 的 supervision 形态

### 后续讨论入口

- 下次继续时优先看:
  - `outputs/refine_v2/full_view_native_stage2a_dual_gpu_sub4_baseline_v1_20260308_0718/videos/final_render.mp4`
  - `outputs/refine_v2/full_view_native_stage2a_fair_v3_20260308_0630`
  - 再决定是推 `Stage 2B`, 还是先做 loss / supervision 调整

## 2026-03-08 09:40 UTC 主题: README 的 `--multi_trajectory` 不是严格可复现入口

### 发现来源

- 对比:
  - `assets/demo/static/diffusion_output_generated/0/rgb/00172.mp4`
  - `assets/demo/static/diffusion_output/0/rgb/00172.mp4`
- 继续追踪:
  - `pose/00172.npz`
  - `intrinsics/00172.npz`
  - `gen3c_single_image_sdg.py`

### 核心问题

- 用户容易把 README 命令理解为:
  - “可以一致还原官方 demo 原版视频”
- 但当前实现里:
  - `--multi_trajectory` 会先随机采样 `movement_distance`
  - 之后才设置 `--seed`
- 结果是:
  - 连相机轨迹本身都不保证复现

### 为什么重要

- 这不是单纯的“像素随机性”
- 而是更上游的 camera contract 已经变化
- 一旦官方 demo 资产又来自不同 `MoGe` / checkpoint / 历史代码环境
- 用户就会误以为模型或权重本身出了问题

### 未来风险

- 如果不把这个事实写清楚:
  - 用户会继续把 `diffusion_output` 当作严格回归基线
  - 但实际上它更像“官方预生成样例”
- 这会让后续的质量回归判断混入:
  - 轨迹变化
  - 内参变化
  - 生成内容变化

### 当前结论

- 已验证:
  - README 当前命令生成的 `00172` 第 `0` 轨迹
  - 与官方 demo 的:
    - `pose`
    - `intrinsics`
    都不一致
- 已验证的直接原因:
  - `random.uniform(...)` 先于 `set_random_seed(...)`
- 尚未完全坐实但很强的候选原因:
  - demo 资产与当前环境的 `MoGe` 版本 / checkpoint / 预处理链不同

### 后续讨论入口

- 如果后面决定修:
  - 先看 `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py`
  - 再补 manifest / README 说明

## 2026-03-08 10:02 UTC 主题: 官方 demo 更接近旧 `MoGe v1`, 当前重新生成更接近 `MoGe v2`

### 发现来源

- 新增 `--moge_version auto|v1|v2` 后
- 在 `00172.png` 上直接跑 `_predict_moge_depth(...)`
- 再对照:
  - `assets/demo/static/diffusion_output/0/intrinsics/00172.npz`
  - `assets/demo/static/diffusion_output_generated/0/intrinsics/00172.npz`

### 核心问题

- 之前我们只能说:
  - 官方 demo 与当前结果的 `intrinsics` 不同
- 现在更进一步看到:
  - 当前重新生成版几乎贴着 `MoGe v2`
  - 官方 demo 则明显更贴近 `MoGe v1`

### 为什么重要

- 这把“demo 差异是不是和 `MoGe` 版本有关”从静态怀疑推进成了动态证据
- 以后再讨论 demo 不一致时:
  - 不该只盯 diffusion 随机性
  - `MoGe` 版本 / checkpoint 也是一等变量

### 未来风险

- 如果 README 继续只给当前命令
- 但不说明 `MoGe v1 / v2` 会改写初始相机内参
- 用户会把:
  - `intrinsics` 差异
  - `trajectory` 差异
  - diffusion 内容差异
  混在一起判断

### 当前结论

- 已验证:
  - `MoGe v1 / v2` 的初始 `w2c` 在该样本上一致
  - 但初始 `intrinsics` 明显不同
  - 当前重新生成版第 `0` 帧内参几乎等于 `v2`
  - 官方 demo 第 `0` 帧内参明显更接近 `v1`
- 仍未完全确认:
  - 官方 demo 是否整条链都固定在旧 `v1` 环境

### 后续讨论入口

- 下一轮优先直接生成一版完整 `--moge_version v1` 对照资产
- 再判断是否还需要继续追别的历史环境差异
