# SPEC: 3D 感知增强（方案 A）v1.0

日期：2026-10-15 ｜ 状态：待评审 ｜ 前置：初始姿态 bug 已修（d31a3c6），像素→3D 链路已验证（base 1.45cm / wrist 0.49cm）

## 0. 背景与目标

失败归因（两轮真实评测 + 探针）：VLM 无法从 2D 像素完成 3D 定位与动作尺度换算，导致 93 步仅 4 步水平移动、原地振荡。

本方案**不改变** VLM 的动作协议（仍是 `[dx,dy,dz,grip]` EE 增量），而是给它的**输入**增加确定性 CV 感知：每步自动把场景中注册对象的精确 3D 位置（世界系、米）写入状态文本。分工：

| 层 | 职责 | 实现 |
|---|---|---|
| VLM | 语义、任务规划、动作决策、避障 | 现有 brain，不动协议 |
| CV 感知 | 对象分割 + 精确 3D 定位 | 新增 `harness/perception.py`（颜色分割+深度反投影，无特权信息、无神经网络） |
| 几何 | 坐标变换 | 已验证的反投影链（内参 + cam2world_gl + GL/CV 轴变换） |
| harness | 有界执行、预算、落盘 | 现有 runner，不动 |

非目标：物体位姿真值（特权信息）绝不进入模型输入；不引入检测/分割神经网络（复杂场景升级时再议）。

## 1. 感知模块 `harness/perception.py`

### 1.1 输入/输出

```python
perceive(sensor_data, sensor_param, task: str) -> list[ObjectReading]

@dataclass
class ObjectReading:
    name: str               # 任务对象表中的名字，如 "cube"
    position: tuple | None  # 世界系 (x,y,z) 米；不可见为 None
    camera: str             # "base_camera" | "hand_camera" | "none"
    mask_pixels: int        # 命中的分割像素数（0 表示不可见）
```

### 1.2 处理管线（每步、每相机）

分割引擎采用 **OpenCV（`opencv-python-headless`，纯 CPU，<10ms/帧）**，不手写图像处理：

1. **HSV 分割**：`cv2.cvtColor(BGR2HSV)` + `cv2.inRange`（红色环回用双区间并集）
2. **形态学清理**：`cv2.morphologyEx(OPEN)` 去噪 + `cv2.connectedComponentsWithStats` 取最大连通域（多对象/同色干扰的工业级实现，MIN_PIXELS=30）
3. **深度采样**：连通域内有效深度（>0 且 <2m）取**中值**；单位 mm→m（CPU 渲染器行为，已实测确认）
4. **双相机融合**：`hand_camera` 像素数 ≥ `WRIST_MIN_PIXELS=200` 时优先用腕部（近距精度高），否则用 `base_camera`；两路都不可见 → `position=None`
5. **反投影**（已验证链路，20 行标准几何，不引 Open3D）：`(u,v,d) → K⁻¹ → GL2CV → cam2world_gl → 世界系`；K、cam2world 从 `sensor_param` 读取（注意 batch 维 `[0]`）

升级路径（不在本期）：复杂开放场景把 §1.2-1 替换为 MobileSAM/YOLO-World（同为 `perceive()` 接口，推理换模型），其余管线不动。

### 1.3 对象注册表 `TASK_OBJECTS`

| 任务 | 注册对象 | 颜色 (RGB) |
|---|---|---|
| PickCube-v1 | cube | 红 (1,0,0) |
| StackCube-v1 | cube_red / cube_green | 红 / 绿 |
| PushCube-v1 / PullCube-v1 | cube | **蓝 (0,0.3,1)**——实现时实测发现这两任务的 cube 是蓝色，且桌面有红色目标标记会污染红分割 |

**PegInsertionSide-v1 不注册**（实现期裁定）：其 peg 渲染为近白色（S≈12），与 panda 本体白色在颜色上不可分，纯色分割无法可靠定位；待分割模型升级后再纳入，当前该任务不输出对象行。

### 1.4 失败语义（写进 state_text，让模型知道）

- 对象不可见 → `not visible`（模型据此产生搜索行为）
- 深度无效（mask 内深度全 0）→ 该路弃用换另一路；两路皆无效 → 同"不可见"
- 任何情况下**不回退到特权真值**，宁可报不可见

## 2. 状态文本增强 `maniskill.py _obs()`

新格式（每对象一行）：

```
end-effector position (m): x=0.012 y=0.038 z=0.182
gripper openness: 1.00 (0=closed, 1=open)
object cube: pos=(x, y, z) m [seen by hand_camera, 512 px]
             delta_to_ee=(dx, dy, dz) m, dist=0.160 m
```

- `delta_to_ee = obj_pos - ee_pos`（世界系）；`dist` = 欧氏距离
- 不可见：`object cube: not visible in either camera`

## 3. VLM 提示词变更 `brain/prompt.py`

SYSTEM_PROMPT 追加三行约定（把上次"尺度靠猜"的坑一并修掉）：

1. 坐标系：状态文本中的位置均为世界系、单位米；`delta_to_ee` 是对象相对末端的位置差
2. 动作尺度：dx/dy/dz ∈ [-1,1]，每步最大位移 0.1m（输出 ±1 → 10cm，±0.5 → 5cm）
3. 夹爪：-1 闭合 / +1 张开；先水平对准（看 delta_to_ee 的 x,y），到位后下降（dz<0）再闭合

输出 JSON 协议**不变**。

## 4. 落盘与报告

- `steps[]` 每步新增 `"perception": {"cube": {"pos": [...], "camera": "...", "mask_pixels": N, "dist": ...}, ...}`
- runner 落盘第三路图 `frames/<集>/obs_overlay/`：在 base/wrist 输入图上画 mask 轮廓 + 质心十字（PIL 画，无 cv2）
- 每集报告卡片新增"3D 感知 (CV)"块：各对象坐标、来源相机、mask 像素数、到 EE 距离
- 逐集 HTML 的输入图替换为叠加版（原图仍存于 obs_base/obs_wrist 不变）

## 5. 文件与体量预算

| 文件 | 变更 | 预计行数 |
|---|---|---|
| `harness/perception.py` | 新增 | ~120 |
| `tests/test_perception.py` | 新增 | ~70 |
| `harness/envs/maniskill.py` | _obs 增强 + 注册表接入 | +25 |
| `harness/brain/prompt.py` | SYSTEM_PROMPT 追加 | +10 |
| `harness/runner.py` | 感知落盘 + overlay | +12 |
| `harness/report.py` | 3D 感知块 + overlay 引用 | +35 |
| `pyproject.toml` | + `opencv-python-headless` | 0 |

总量预计 +270 行 → 项目 ~1300 行，符合 ≤2000 行约束。

## 6. 验收标准与实测结果（1-5 全部离线，零 API 成本）

1. **反投影精度**：方块真值 vs CV 位置，base ≤2cm、wrist ≤1cm（实测基线 1.45/0.49cm）✅
2. **检出与鲁棒**：12 格（PegInsertion 不注册）× 5 seeds，"可见(≥8px)即检出"。**实测 74/75**（唯一 miss 为 panda/PickCube seed3 方块仅 4px，属真实不可见），**平均误差 2.01cm、最大 2.53cm** ✅
3. **无撞色**：StackCube 双方块互不误检（cube_red/cube_green 各自独立检出）；Push/Pull 的红色目标标记经蓝色注册规避 ✅
4. **回归**：21 测试全绿（新增 `test_perception.py` 6 项：合成分割、反投影精度、腕部优先、state 格式、JSON 序列化）✅
5. **单集冒烟**：random brain 1 集全链路跑通，报告卡片含 3D 感知块与叠加图 ✅

实现期修正记录（对 §1.2 的偏离）：① 分割后**不做形态学开运算**——3×3 opening 会抹掉 8-12px 的最小合法检出；② 新增**世界系高度过滤**（组件反投影后 z>0.30m 判为机器人本体色，剔除 fetch 腕部蓝饰带这类干扰）；③ MIN_PIXELS=8（12 会漏掉远角 11px 的方块）。

## 7. 在线验证（验收通过后，需 API）

- 2 集 PickCube × panda（60 步上限）修复后基线 vs 3D 增强，逐项对比：成功率 / 行为分段 / 指点-执行归因
- 预算封顶 40 万 token；成功即停

## 8. 风险与边界

- **HSV 对光照敏感**：lavapipe 渲染光照固定，风险低；换 GPU 渲染或真实相机时需重标定（记录在 perception.py 模块注释）
- **同色多实例**：当前任务表保证无此情况；未来场景需升级为连通域/检测模型（明确不在本期）
- **深度边缘误差**：mask 中值法对立方体侧面深度混合不敏感（已实测 0.49cm），但斜视角远距会退化——腕部优先策略缓解
- **初始姿态修复仅覆盖 panda_wrist**：fetch/xarm6 走 ManiSkill 原生分支无此问题；未来新增自定义 uid 时必须同步 `_fix_custom_uid_init`
