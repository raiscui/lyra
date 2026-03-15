# `multi_trajectory` 相机运动实现说明

## 1. 这份文档是做什么的

这份文档把当前项目里 `--multi_trajectory` 的实现方式拆开讲清楚.
目标不是只说明"代码在哪", 而是把这套设计整理成其他项目可以直接借鉴的方案.

如果你只想先抓主线, 可以先记住一句话:

- 上层 `demo_multi_trajectory(args)` 负责枚举 6 条轨迹, 分配 `traj_idx`, 并为每条轨迹采样 `movement_distance`.
- 底层 `generate_camera_trajectory(...)` 负责把轨迹名字真正转换成逐帧相机位姿序列.

相关流程图见:

- [specs/2026-03-15-multi-trajectory-camera-flow.md](../specs/2026-03-15-multi-trajectory-camera-flow.md)

---

## 2. 当前项目里的代码位置

### 2.1 `multi_trajectory` 参数入口

静态单图 SDG 入口:

- `scripts/bash/static_sdg.sh`
- `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py:126-129`

动态视频 SDG 入口:

- `scripts/bash/dynamic_sdg.sh`
- `cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py:120-123`

### 2.2 6 条轨迹的枚举位置

静态版:

- `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py:1023-1043`
- 关键函数: `demo_multi_trajectory(args)`

动态版:

- `cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py:870-912`
- 关键函数: `demo_multi_trajectory(args)`

### 2.3 底层相机轨迹生成位置

统一走:

- `cosmos_predict1/diffusion/inference/camera_utils.py:151-246`
- 函数: `generate_camera_trajectory(...)`

它继续分发到两个底层实现:

- 平移轨迹: `create_horizontal_trajectory(...)`
  - `cosmos_predict1/diffusion/inference/camera_utils.py:48-97`
- 环绕轨迹: `create_spiral_trajectory(...)`
  - `cosmos_predict1/diffusion/inference/camera_utils.py:100-148`

---

## 3. 这套设计的核心思路

这套实现实际上分成两层.

### 第 1 层. 轨迹枚举层

这一层不负责数学轨迹本身.
它只做 4 件事:

1. 定义有哪些轨迹名字.
2. 给每条轨迹分配固定 `traj_idx`.
3. 给每条轨迹定义一个 `movement_distance_range`.
4. 逐条调用主 `demo(args)` 流程.

也就是说, `multi_trajectory` 不是一次生成一个复杂大轨迹.
它是把多个单轨迹任务拆开, 一条一条分别生成.

### 第 2 层. 轨迹几何层

这一层由 `generate_camera_trajectory(...)` 负责.

它根据 `trajectory_type` 做分发:

- `left/right/up/down/zoom_in/zoom_out` -> 线性平移轨迹
- `clockwise/counterclockwise` -> 螺旋/绕拍轨迹

这一层只关心:

- 相机沿哪条路径移动
- 每一帧的位置怎么计算
- 相机朝向怎么更新
- 最终输出 `generated_w2cs` 和 `generated_intrinsics`

这个分层很值得借鉴.
因为以后你要新增轨迹时, 往往只需要改其中一层:

- 新增一种批量策略 -> 改枚举层
- 新增一种几何轨迹 -> 改几何层

---

## 4. 当前项目实际启用的 6 条轨迹

当前 `multi_trajectory` 实际启用的是下面 6 条.
不是 7 条.
注意: 底层虽然支持 `down`, 但当前枚举层没有把 `down` 放进 `multi_trajectory`.

| 轨迹名 | `traj_idx` | `movement_distance_range` | 底层类型 | 说明 |
| --- | ---: | --- | --- | --- |
| `left` | 0 | `[0.2, 0.3]` | 线性平移 | 相机沿 `x` 负方向平移 |
| `right` | 1 | `[0.2, 0.3]` | 线性平移 | 相机沿 `x` 正方向平移 |
| `up` | 2 | `[0.1, 0.2]` | 线性平移 | 相机沿 `y` 负方向平移 |
| `zoom_out` | 3 | `[0.3, 0.4]` | 线性平移 | 相机沿 `z` 负方向后退 |
| `zoom_in` | 4 | `[0.3, 0.4]` | 线性平移 | 相机沿 `z` 正方向前进 |
| `clockwise` | 5 | `[0.4, 0.6]` | 螺旋环绕 | 相机围绕目标做顺时针 orbit |

额外固定参数:

```python
camera_gen_kwargs = {
    "radius_x_factor": 0.15,
    "radius_y_factor": 0.10,
    "num_circles": 2,
}
```

这组参数主要影响 `clockwise` 这种螺旋轨迹.

---

## 5. `demo_multi_trajectory(args)` 这一层到底怎么做

### 5.1 它的行为可以概括成下面的伪代码

```python
from random import uniform
from pathlib import Path

TRAJECTORIES = {
    "left": {"traj_idx": 0, "movement_distance_range": [0.2, 0.3]},
    "right": {"traj_idx": 1, "movement_distance_range": [0.2, 0.3]},
    "up": {"traj_idx": 2, "movement_distance_range": [0.1, 0.2]},
    "zoom_out": {"traj_idx": 3, "movement_distance_range": [0.3, 0.4]},
    "zoom_in": {"traj_idx": 4, "movement_distance_range": [0.3, 0.4]},
    "clockwise": {"traj_idx": 5, "movement_distance_range": [0.4, 0.6]},
}

CAMERA_GEN_KWARGS = {
    "radius_x_factor": 0.15,
    "radius_y_factor": 0.10,
    "num_circles": 2,
}


def demo_multi_trajectory(args):
    root = args.video_save_folder
    args.camera_gen_kwargs = dict(CAMERA_GEN_KWARGS)

    for traj_name, meta in TRAJECTORIES.items():
        args.video_save_folder = str(Path(root) / str(meta["traj_idx"]))
        args.trajectory = traj_name
        args.movement_distance = (
            uniform(*meta["movement_distance_range"])
            * args.total_movement_distance_factor
        )
        demo(args)
```

### 5.2 这层设计为什么实用

因为它天然解决了 3 个问题:

1. 输出组织问题
   - 每条轨迹都有自己的子目录, 不会互相覆盖.
2. 数据多样性问题
   - 同一种轨迹可以在一个区间里随机采样强度, 避免全是同一个运动幅度.
3. 训练和推理复用问题
   - 轨迹枚举逻辑不必和底层渲染/扩散耦死.

### 5.3 如果你是其他项目, 最值得照抄的点

- 用 `traj_idx` 固定输出目录.
- 不要把所有轨迹混在一个大循环输出到同一层.
- 不要把轨迹强度写死成单值. 用区间采样更稳.
- 把批量轨迹枚举和底层轨迹生成拆开.

---

## 6. `generate_camera_trajectory(...)` 这一层怎么做分发

底层入口函数的核心职责是:

- 根据 `trajectory_type` 选择轨迹族
- 调对应的几何生成函数
- 把结果整理成统一的 batch 维度

可以理解成下面的伪代码:

```python
def generate_camera_trajectory(
    trajectory_type,
    initial_w2c,
    initial_intrinsics,
    num_frames,
    movement_distance,
    camera_rotation,
    center_depth=1.0,
    translation_reference_depth=None,
    num_circles=1,
    radius_x_factor=1.0,
    radius_y_factor=1.0,
):
    if trajectory_type in ["clockwise", "counterclockwise"]:
        radius_x = movement_distance * radius_x_factor
        radius_y = movement_distance * radius_y_factor
        w2cs = create_spiral_trajectory(...)
    else:
        axis, positive = map_linear_trajectory(trajectory_type)
        w2cs = create_horizontal_trajectory(...)

    generated_w2cs = w2cs.unsqueeze(0)
    generated_intrinsics = repeat_intrinsics(initial_intrinsics, num_frames)
    return generated_w2cs, generated_intrinsics
```

这个接口设计也很值得借鉴.
因为调用方只需要传轨迹名字.
它不需要知道每种轨迹背后的数学细节.

---

## 7. 6 条轨迹各自的实现方式

这一节是最适合其他项目直接抄思路的部分.

### 7.1 `left`

映射规则:

- `axis = "x"`
- `positive = False`

位移公式:

```python
x = i * distance * translation_depth / n_steps * (-1)
y = 0
z = 0
```

含义:

- 相机逐帧向左平移.
- 如果 `camera_rotation == "center_facing"`, 相机会一直看向同一个中心点.

### 7.2 `right`

映射规则:

- `axis = "x"`
- `positive = True`

位移公式:

```python
x = i * distance * translation_depth / n_steps * (+1)
y = 0
z = 0
```

含义:

- 相机逐帧向右平移.

### 7.3 `up`

映射规则:

- `axis = "y"`
- `positive = False`

位移公式:

```python
x = 0
y = i * distance * translation_depth / n_steps * (-1)
z = 0
```

含义:

- 相机逐帧向上移动.
- 当前实现里默认约定 `y` 正方向更接近 "down".
  所以 `up` 用的是负方向.

### 7.4 `zoom_out`

映射规则:

- `axis = "z"`
- `positive = False`

位移公式:

```python
x = 0
y = 0
z = i * distance * translation_depth / n_steps * (-1)
```

含义:

- 相机沿着视线方向后退.
- 常用于做拉远镜头.

### 7.5 `zoom_in`

映射规则:

- `axis = "z"`
- `positive = True`

位移公式:

```python
x = 0
y = 0
z = i * distance * translation_depth / n_steps * (+1)
```

含义:

- 相机沿着视线方向前进.
- 常用于做推镜头.

### 7.6 `clockwise`

映射规则:

- 走 `create_spiral_trajectory(...)`
- `positive = True`
- `radius_x = movement_distance * radius_x_factor`
- `radius_y = movement_distance * radius_y_factor`
- `num_circles = 2` in current multi-trajectory preset

位移公式:

```python
theta = theta_max * i / (n_steps - 1)
x = radius_x * (cos(theta) - 1) * translation_depth
y = radius_y * sin(theta) * translation_depth
z = radius_z * sin(theta) * translation_depth
```

含义:

- 相机会围绕目标做顺时针的螺旋/环绕运动.
- 因为当前 preset 给了 `radius_x_factor=0.15` 和 `radius_y_factor=0.10`, 所以它不是一个标准正圆, 更接近横向幅度稍大的椭圆轨迹.
- `num_circles=2` 表示整个片段里转 2 圈.

---

## 8. 相机朝向是怎么控制的

轨迹只有位置还不够.
实际镜头效果还取决于朝向更新策略.
当前项目支持 3 种模式:

- `center_facing`
- `trajectory_aligned`
- `no_rotation`

### 8.1 `center_facing`

相机始终看向固定的 `look_at = [0, 0, center_depth]`.

适合:

- 物体居中展示
- 产品转台
- 人像主体尽量保持居中

### 8.2 `trajectory_aligned`

相机朝向轨迹的前进方向.
当前实现方式是:

```python
look_at = base_look_at + pos * 2
```

这不是严格的切线追踪.
但它足够轻量, 效果上接近"镜头朝着运动方向走".

### 8.3 `no_rotation`

相机随位置移动, 但尽量不额外做转向补偿.
当前实现里是:

```python
look_at = base_look_at + pos
```

效果上更接近相机平移, 而不是持续盯住中心点.

---

## 9. 为什么这里有 `center_depth` 和 `translation_reference_depth`

这是当前实现里一个很关键, 也很容易被别的项目忽略的设计点.

### `center_depth`

它决定的是:

- 镜头主要看向哪里
- `look_at` 的深度基准是什么

### `translation_reference_depth`

它决定的是:

- 实际位移量按多大的深度尺度放大

也就是说, 当前实现把"看向哪里"和"移动多远"拆开了.
这个设计很值得借鉴.

否则常见的问题是:

- 主体看向调对了, 但位移突然过大
- 或者位移强度看起来对了, 但镜头盯的位置不对

如果你的场景深度变化很大, 强烈建议保留这两个参数的解耦.

---

## 10. 给其他项目的最小可移植版本

如果你不想把这个仓库整套搬过去, 最小可以只移植下面 3 层.

### 第 1 层. 轨迹配置表

```python
TRAJECTORIES = {
    "left": {"traj_idx": 0, "range": [0.2, 0.3]},
    "right": {"traj_idx": 1, "range": [0.2, 0.3]},
    "up": {"traj_idx": 2, "range": [0.1, 0.2]},
    "zoom_out": {"traj_idx": 3, "range": [0.3, 0.4]},
    "zoom_in": {"traj_idx": 4, "range": [0.3, 0.4]},
    "clockwise": {"traj_idx": 5, "range": [0.4, 0.6]},
}
```

### 第 2 层. 名字到几何轨迹的映射器

```python
def map_linear_trajectory(name: str) -> tuple[str, bool]:
    if name == "left":
        return "x", False
    if name == "right":
        return "x", True
    if name == "up":
        return "y", False
    if name == "down":
        return "y", True
    if name == "zoom_in":
        return "z", True
    if name == "zoom_out":
        return "z", False
    raise ValueError(f"unsupported trajectory: {name}")
```

### 第 3 层. 两类轨迹生成器

- 线性平移生成器
- 螺旋环绕生成器

这已经足够支撑大多数:

- 单物体展示视频
- 静态图生视频镜头运动
- 3DGS / NeRF / Gaussian Splatting 的多视角监督采样
- 轻量级训练数据增强

---

## 11. 其他项目接入时, 我建议你优先保留的设计

### 建议 1. 轨迹名字和数学轨迹解耦

不要在业务代码里到处写 `if name == "left"`.
集中到一个分发器里.

### 建议 2. 枚举层和几何层解耦

不要让 `demo_multi_trajectory` 直接充满数学公式.
否则你后面新增批量策略会很痛苦.

### 建议 3. 每条轨迹单独输出目录

这个设计对训练数据和断点续跑都很友好.

### 建议 4. 用区间采样代替固定强度

尤其是训练场景.
同一个轨迹固定同一个 `movement_distance` 往往太僵硬.

### 建议 5. 保留 `camera_rotation` 这个维度

只有位移没有朝向, 你得到的镜头语言会很死.

### 建议 6. 保留 `center_depth` 和 `translation_reference_depth` 的解耦

这是这套实现里最容易被低估, 但非常实用的一点.

---

## 12. 一个更适合别的项目直接抄走的参考结构

```python
class TrajectoryPreset(TypedDict):
    traj_idx: int
    movement_distance_range: tuple[float, float]
    family: str


TRAJECTORY_PRESETS: dict[str, TrajectoryPreset] = {
    "left": {"traj_idx": 0, "movement_distance_range": (0.2, 0.3), "family": "linear"},
    "right": {"traj_idx": 1, "movement_distance_range": (0.2, 0.3), "family": "linear"},
    "up": {"traj_idx": 2, "movement_distance_range": (0.1, 0.2), "family": "linear"},
    "zoom_out": {"traj_idx": 3, "movement_distance_range": (0.3, 0.4), "family": "linear"},
    "zoom_in": {"traj_idx": 4, "movement_distance_range": (0.3, 0.4), "family": "linear"},
    "clockwise": {"traj_idx": 5, "movement_distance_range": (0.4, 0.6), "family": "spiral"},
}
```

这样后面你要新增:

- `counterclockwise`
- `down`
- `arc_left`
- `dolly_zoom`

都会更顺手.

---

## 13. 当前实现和可扩展点

### 当前实现已经有的能力

- 支持多轨迹批量生成
- 支持平移和环绕两大家族
- 支持相机朝向模式切换
- 支持 `movement_distance` 区间采样
- 支持把 pose/intrinsics 随轨迹一起导出
- 动态版支持 `flip_supervision`

### 当前实现没放进 `multi_trajectory` 但底层已支持的能力

- `down`
- `counterclockwise`
- `no_rotation`
- `trajectory_aligned`

如果你的项目需要更多镜头变化, 这几个其实已经是现成扩展口.

---

## 14. 最后给一个落地建议

如果你要在别的项目里照搬, 我建议按下面顺序落地:

1. 先移植 `generate_camera_trajectory(...)` 的接口形状.
2. 再移植 `create_horizontal_trajectory(...)` 和 `create_spiral_trajectory(...)`.
3. 最后再加 `demo_multi_trajectory(...)` 这一层批量枚举器.

原因很简单:

- 底层几何层先稳定, 才好验证镜头运动对不对.
- 上层批量枚举层晚一点接, 才不会一上来把问题放大成 6 条轨迹一起调不明白.

如果你只想快速复现当前项目的行为, 那就直接照抄:

- 6 条轨迹名字
- 对应 `traj_idx`
- 对应 `movement_distance_range`
- `radius_x_factor=0.15`
- `radius_y_factor=0.10`
- `num_circles=2`

这样最接近当前仓库的 `multi_trajectory` 效果.
