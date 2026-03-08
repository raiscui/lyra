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

## 2026-03-08 12:09 UTC 主题: `MoGe v2` 的空间感异常更像 FOV 变化, 不像 pose 错乱

### 发现来源

- 用户指出:
  - `MoGe v1` 输出透视正常
  - `MoGe v2` 输出空间感不正常
- 随后对照了:
  - 官方 demo
  - 当前 `v1` 生成版
  - 当前 `v2` 生成版
- 并回读了:
  - `moge/model/v1.py`
  - `moge/model/v2.py`
  - `gen3c_single_image_sdg.py`

### 核心问题

- 当画面空间感明显变弱时, 第一反应很容易怀疑:
  - 相机轨迹错了
  - 或 `intrinsics` 契约用错了
- 但这次更强的证据是:
  - 姿态差异很小
  - 焦距和 FOV 差异很大

### 为什么重要

- 这意味着后续判断空间感问题时:
  - 不能只看 `pose`
  - 也不能只看“视频看起来怪”
  - 要把 `intrinsics / FOV` 当成一等变量
- 尤其在:
  - `MoGe v1 / v2`
  - 不同 checkpoint
  - 不同 demo 资产
 之间切换时

### 未来风险

- 如果后面继续把:
  - `MoGe` 版本变化
  - `multi_trajectory` 的随机位移
  - diffusion 内容差异
  混在一起讨论
- 很容易把“视场变化导致的视差变弱”误判成“轨迹错乱”或“代码回退”

### 当前结论

- 已验证:
  - `v1 / v2` 的姿态非常接近
  - `v2` 的 `fx / fy` 更小
  - `v2` 的 `FOV` 更大
  - `demo` 更接近 `v1`
- 当前最合理的判断是:
  - `v2` 输出更扁, 更像是 FOV 变广带来的自然结果
  - 不是 `pose` 大幅偏了
  - 也还没有证据表明是 Lyra 把 `v2 intrinsics` 用坏了

### 后续讨论入口

- 若要再增强证据:
  - 做一次固定 `trajectory + movement_distance` 的 `v1 / v2` 单轨迹 A/B
- 若要改用户体验:
  - README 或命令行应更明确提示:
    - `auto` 默认会走 `v2`
    - 若想更接近旧 demo, 优先试 `--moge_version v1`

## 2026-03-08 12:47 UTC 主题: 这次更像“v2 focal calibration”问题, 不是“camera path redesign”问题

### 发现来源

- 在继续追问“怎样把 `v2` 调回接近 `v1`”时
- 对比了:
  - `fx_v1 / fx_v2`
  - `fy_v1 / fy_v2`

### 核心问题

- 很多时候看到空间感异常, 容易本能去改:
  - `movement_distance`
  - 轨迹形状
  - 相机旋转策略
- 但这次数值特征更像:
  - `v2` 的 `fx/fy` 整体偏小一个稳定比例

### 为什么重要

- 如果这是“全局 focal scale”问题
- 最干净的修法就是:
  - 只改 `fx/fy`
  - 不动 `cx/cy`
  - 不动 pose
- 这比把路径也卷进来要可控得多

### 当前结论

- 已验证:
  - `fx_v1 / fx_v2 = 1.0463717534`
  - `fy_v1 / fy_v2 = 1.0463716675`
  - 两者几乎完全一致
- 因此当前最合理的工程化做法是:
  - 提供显式 `v2 focal correction` 开关
  - 而不是先改 path

### 后续讨论入口

- 等 `diffusion_output_generated_v2_focalfix` 有首批产物后
- 优先先看:
  - “仅改 focal”能否已经把空间感拉回
- 只有这条还不够时
- 再讨论 `movement_distance` 补偿或固定轨迹 A/B

## 2026-03-08 13:12 UTC 主题: focal correction 的视觉收益会被随机轨迹幅度直接抵消

### 发现来源

- 用户反馈:
  - `v2 focal fix` 看起来变化不大
- 随后对比了现有第 `0` 条轨迹的:
  - `v2_orig`
  - `v2_focalfix`
  - `v1`

### 核心问题

- 只看内参是否拉回, 还不够
- 如果同一轮的 `movement_distance` 变小了
- 那么更大的焦距带来的视差增强, 会被更短的路径直接抵消

### 为什么重要

- 这说明:
  - `focal correction` 和 `trajectory amplitude`
  - 不是两个可以分开主观评估的独立变量
- 一旦 `multi_trajectory` 仍在随机采样轨迹强度
- 视觉 A/B 就会持续被污染

### 当前结论

- 已验证:
  - `v2_focalfix` 的内参已几乎等于 `v1`
  - 但它的 `path_len` 更短
  - 所以 `path * fx` 几乎没变
- 因而:
  - “变化不大”目前更像随机轨迹抵消
  - 不是 focal correction 无效

### 后续讨论入口

- 若要干净验证 `focal correction`:
  - 必须固定:
    - `trajectory`
    - `movement_distance`
  - 最好先禁用 `multi_trajectory`

## 2026-03-08 13:35 UTC 主题: 放大 movement 但保持固定 `center_depth=1.0` 会把转轴视觉上推到前方

### 发现来源

- 用户对 `v2_move2` 的直接反馈:
  - “相机的转轴很靠前”
- 随后回读:
  - `camera_utils.py`
  - `gen3c_single_image_sdg.py`
  并对照 `v1 / v2_move2` 的 pose 数值

### 核心问题

- 当前轨迹生成不是围绕“场景真实中心”旋转
- 而是围绕一个固定的虚拟盯视点:
  - `[0, 0, center_depth]`
- 在当前脚本里:
  - `center_depth` 被硬编码成 `1.0`

### 为什么重要

- 这意味着:
  - 一旦只把 `movement_distance` 往上加
  - 而不同时调整 `center_depth`
  - 运动会越来越像“绕近处点甩头”
- 用户感受到的“转轴靠前”
  - 很可能就是这个固定旋转中心带来的

### 当前结论

- 已验证:
  - `v2_move2` 的 `path_len` 已达到 `1.80x v1`
  - 但旋转中心深度仍固定 `1.0`
- 因而:
  - `move2` 更像在放大“绕近点转”的程度
  - 而不是单纯把原来 `v1` 的路径等比例放大

### 后续讨论入口

- 如果下一步要让轨迹更自然:
  - 不是继续无脑加 `movement_distance`
  - 而是考虑让 `center_depth` 也可调
  - 或者直接从 MoGe 深度统计里估计一个更合理的 rotation center depth

## 2026-03-08 14:46 UTC 主题: 固定 `center_depth=1.0` 在 `v2` 尺度下过于靠前

### 发现来源

- 用户指出:
  - 原始 `v2` 版本也觉得转轴靠前
- 随后直接对 `00172.png` 运行了:
  - `MoGe v1`
  - `MoGe v2`
  的深度推理

### 核心问题

- 我们之前已经知道:
  - 轨迹代码固定看向 `z=1.0`
- 这次新的关键证据是:
  - `v2` 的场景深度整体比 `v1` 远很多

### 为什么重要

- 这意味着:
  - 同一个 `center_depth=1.0`
  - 在 `v1` 和 `v2` 里并不代表“同样相对位置”的旋转中心
- 所以即使 path 差不多
- `v2` 也会天然更像在围着前景点转

### 当前结论

- 已验证:
  - `v1 center_median ≈ 3.16`
  - `v2 center_median ≈ 13.60`
  - 固定 `center_depth=1.0` 在 `v2` 中相对场景明显过近
- 因而:
  - 原始 `v2` 的“转轴靠前”是结构性问题
  - `move2` 只是把它继续放大

### 后续讨论入口

- 如果要真正修:
  - 优先做 `center_depth` 可调
  - 更进一步则用 MoGe 深度统计自动估计 rotation center depth

## 2026-03-08 13:13 UTC 主题: `Stage 2A` 不是几何修复阶段, baseline 空间厚化要先回到初始化链路看

### 发现来源
- 用户追问 `sample.py` 导出的 baseline 高斯是否真的考虑了 depth / point cloud / `VIPE`.
- 本轮顺着 `sample.py -> model_latent_recon -> refinement_v2` 调用链做了静态与最小动态验证.

### 核心问题
- 很容易把“训练时模型见过 depth loss”误解成“当前这次 baseline 重建会实时吃当前场景 depth”.
- 也容易把 `Stage 2A` 当成一个会自动修正空间几何的阶段.
- 这两个理解在当前仓库里都不成立.

### 为什么重要
- 这决定了后续排查空间拉丝/厚表面/沿视线方向拉长时,第一落点应该在哪里.
- 如果一开始就把问题归到 `Stage 2A`, 很容易在错误阶段上反复调 appearance loss, 但几何不会真正变好.

### 未来风险
- 如果团队后面继续把 `Stage 2A` 的视觉提升等同于几何提升, 会在评估时混淆:
  - baseline 初始化误差
  - observation 稀疏带来的深度歧义
  - appearance cleanup 带来的“看起来更干净”
- 这样会导致参数调优方向跑偏, 花很多算力但仍解决不了空间厚化.

### 当前结论
- 当前已知事实:
  1. `sample.py` 用户这条命令运行时 `use_depth = false`
  2. baseline 高斯中心来自网络预测的沿 ray 深度回投
  3. `Stage 2A` 冻结 `means` / `rotations`
  4. `VIPE` 不在 static baseline 路线里
- 仍未确认的部分:
  - 具体某个场景里的拉丝主要占比是 baseline 初始化误差, 还是 observation 不足, 还需要对真实 `.ply` 和多角度 render 做专门验证

### 后续讨论入口
- 如果下一轮要真的解决这类空间错误, 优先看:
  - `sample.py`
  - `src/models/recon/model_latent_recon.py`
  - `src/refinement_v2/gaussian_adapter.py`
  - `src/refinement_v2/runner.py` 的 `stage2b`

## 2026-03-08 13:29 UTC 主题: 不要把 demo 生成链路里的 `MoGe` 使用自动等同于官方训练 depth 的来源

### 发现来源
- 用户追问训练数据里的 depth 是否离线由 `MoGe` 生成.
- 本轮沿 `dataset registry -> Radym dataloader -> README training section -> SDG generator` 做了完整追踪.

### 核心问题
- 仓库里同时存在两类事实:
  1. 单图 / SDG 推理链路明确会调用 `MoGe`
  2. 训练 dataloader 明确会读取现成 `depth/*.zip`
- 但这两件事之间没有在仓库中被正式连成“训练 depth = MoGe 输出”.

### 为什么重要
- 这类口径如果说快了, 很容易把“强相关”讲成“已确认因果”.
- 后续一旦有人根据这句话做复现实验或替换 depth 生成器, 就会踩到证据断层.

### 未来风险
- 如果不把这条边界记下来:
  - 团队后续会继续把官方训练集 depth 的 provenance 说得过满
  - 也会把 demo `generated` 目录没有 depth 的现象误读成“训练其实不用 depth”

### 当前结论
- 当前已知事实:
  1. dataloader 读取 `depth/*.zip`
  2. static SDG 脚本用 `MoGe` 估初始深度/内参
  3. static SDG 默认不落盘训练 supervision depth
- 当前未知事实:
  - 官方 ModelScope 训练集里的 depth 是否由 `MoGe`、`Depth Anything` 或别的内部流程生成

### 后续讨论入口
- 如果下次还要继续追 provenance,优先看:
  - ModelScope 数据集页面 / 配套文档
  - 论文 supplementary / data generation section
  - 仓库外的数据准备脚本(若后续补到仓库)
