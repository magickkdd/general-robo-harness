# 项目交接文档

日期：2026-10-15 ｜ 版本：harness v1.0（commit 650f5dc + 后续） ｜ 维护者交接用

## 1. 项目是什么

通用**跨本体、跨 VLM 大脑**的具身评测 harness。两条可插拔的轴：

- **环境轴**：ManiSkill3，任务 × 本体矩阵（panda / xarm6_robotiq / fetch × 5 桌面任务，13 个合法组合），统一 EE 增量动作协议 `[dx, dy, dz, grip]`（[-1,1]，1.0 = 10cm/步）
- **大脑轴**：任意 OpenAI 兼容 API（vision 模型），图像（第三人称 + 腕部）+ 本体状态文本 → 严格 JSON 动作

当前核心成果：**3D 感知增强**（方案 A）——OpenCV 经典 CV 从 RGBD 反投影出对象级世界系坐标，写进 VLM 的状态文本。实测把模型行为从"原地振荡"变成"10 步内精确接近到 3.3cm"。

规模：~1250 行 Python、12 个源文件、21 个测试（全绿）。

## 2. 环境与启动

```bash
# Python 环境：项目根 .venv（Python 3.10，uv 创建）
source ~/.bashrc  # 无需；直接用 .venv/bin/python 或 .venv/bin/harness

# 依赖已装好；重建时（国内镜像）：
uv venv --python /usr/bin/python3.10 .venv
uv pip install --python .venv/bin/python torch==2.14.1+cpu \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/
uv pip install --python .venv/bin/python -e ".[sim,dev]" \
  --index-url https://mirrors.aliyun.com/pypi/simple/
```

关键环境事实（代码已自动处理，重建环境时需知晓）：

- **Vulkan**：本机无 NVIDIA ICD，`harness/envs/maniskill.py` 启动前自动设 `VK_ICD_FILENAMES` 指向 lavapipe 软渲染（CPU，~0.05s/帧）
- **import 顺序铁律**：`import harness.envs` 必须先于任何 `import sapien`——sapien 导入时会探测 Vulkan 并缓存失败实例
- **深度单位**：CPU 渲染器的深度是**整数毫米**（perception 里 /1000）
- **机器人资产**：`~/.maniskill/data/robots/`（已下载 xarm6/widowxai/ur10e；HF 走 hf-mirror）
- **密钥**：`.env`（HARNESS_VLM_KEY=deepseek-v4-flash-vision 网关；LLM_API_KEY/DEEPSEEK_API_KEY/SENSENOVA_API_KEY 为旧 key）

## 3. 代码地图（12 个源文件）

| 文件 | 职责 | 关键位置 |
|---|---|---|
| `harness/envs/base.py` | 规范协议：Obs/StepResult/TaskCell/EnvAdapter | 全文 60 行，系统的"宪法" |
| `harness/envs/maniskill.py` | ManiSkill 适配器 | `_fix_custom_uid_init`（L139，**重要 bugfix**）；`_build_layout`（fetch 9 维动作映射）；`_obs`（感知接入） |
| `harness/envs/__init__.py` | 后端注册表 | 加 LIBERO 就在这里 |
| `harness/perception.py` | **3D 感知**：OpenCV 分割+反投影 | `_HSV_RANGES`（实测标定）；`TASK_OBJECTS`（按任务注册对象颜色）；`MAX_OBJECT_Z_M=0.30` 高度过滤 |
| `harness/brain/prompt.py` | 提示词 + 动作 JSON 解析 | `parse_action_json`（截断自动补 `}`）；调 prompt 就改这里 |
| `harness/brain/vlm_api.py` | OpenAI 兼容客户端 | `_chat`（双图+reasoning 捕获）；`act`（1 次修复重试+零动作兜底） |
| `harness/brain/__init__.py` | RandomBrain + 工厂 | 冒烟测试用 random，零成本 |
| `harness/runner.py` | episode 循环 | `Budget`（episodes/calls/tokens 三重封顶）；`run_episode`；`encode_video` |
| `harness/metrics.py` / `report.py` | 聚合 + 报告 | 逐集 HTML 内嵌每步完整管线 |
| `harness/cli.py` | CLI 入口 | `harness list` / `harness run` |

测试：`tests/` 6 个文件（快测 + `test_integration.py` 真仿真慢测，`-m "not slow"` 跳过）。

## 4. 怎么跑

```bash
.venv/bin/harness list                    # 能力矩阵
.venv/bin/harness run --task PickCube-v1 --embodiment panda --brain random \
  --episodes 1                            # 零成本冒烟
.venv/bin/harness run --task PickCube-v1 --embodiment panda --brain vlm \
  --model deepseek-v4-flash-vision \
  --base-url https://discovery-api.intern-ai.org.cn/v1 \
  --api-key-env HARNESS_VLM_KEY \
  --episodes 1 --max-steps 60 --record-video \
  --max-brain-calls 250 --max-tokens 1500000 \
  --out runs/<名字>
.venv/bin/python -m pytest tests/ -q      # 全部测试（含 1 个慢测 ~15s）
```

产物 `runs/<名字>/`：`episodes/*.json`（逐步轨迹：动作/奖励/state_text/perception/完整 reasoning/模型原话）、`frames/`（obs_base、obs_wrist、obs_overlay 分割叠加、render 帧）、`videos/*.mp4`、`summary.json`、`report.html`、`episode_*.html`（每步完整管线页面）。

**看报告**：同步到 Windows D 盘 `cp runs/<名>/{report.html,summary.json,episode_*.html} /mnt/d/harness-reports/<名>/ && cp runs/<名>/videos/*.mp4 ...`（本机是 WSL，D 盘挂 /mnt/d）。

## 5. 实验状态与结论（截至交接）

已验证/已修的关键坑（按发现顺序）：

1. **模型回复截断**：deepseek-v4 思考 ~1200 token，max_tokens=512 必截断 → 解析器截断补 `}`；2026-10-15 实测思考可达 ~3900 字符，**2048 也会被吃满 → content 空串**（57/60 步空回复、每步 2 次调用、34 步零动作）→ 默认提到 4096
2. **reasoning_content**：思考在 API 的独立字段；现已完整落盘（勿再截断——35/47 步撞过 2000 字上限）
3. **思考不可信**：模型会幻觉引用自己输入的数字，核对以落盘的图像+state_text 为准
4. **panda_wrist 初始姿态 bug**（重要）：ManiSkill TableSceneBuilder 按 uid 硬编码初始姿态，自定义 uid 会关节全零出生 → `_fix_custom_uid_init` 修复。**新增自定义 uid 必须同步此处**。此 bug 前的评测数据全部作废
5. **3D 感知验收**（12 格 × 5 seeds）：74/75 检出、平均误差 2.01cm / 最大 2.53cm（`runs/perception_acceptance.json`）
6. **感知实现三坑**：① 不做形态学开运算（抹掉 8-12px 最小检出）② 世界系高度过滤 z>0.30m 剔机器人本体色（fetch 腕部蓝饰带）③ MIN_PIXELS=8
7. **PushCube/PullCube 的 cube 是蓝色**且桌面有红色目标标记（红分割会被带偏）；**PegInsertionSide 未注册**（白色 peg 与白色机身颜色不可分，待换学习型分割）

最新一轮（3D 增强，`runs/vlm_3d_check/`）：EE→方块距离 0.14→0.033m（step 10，方案 A 实锤有效），**但** step 10 后夹爪自遮挡 base 视线 + 腕部太近失明 → 45 步"失明"漂移；修复重试 ~40 次/集（首轮失败原因未记录，是可观测性缺口）。两集均未成功，第 2 集被全局 call 预算止损。

2026-10-15/16 StackCube 系列调试（均 `--gt` 特权真值调试模式，不进评测；接口冻结原则在此期间确立，见 §8）：
- **auto_cruise 步幅策略已回退**（876ad03 引入、56b493e 撤销）：harness 放大/截停模型输出违反接口冻结——模型输出什么就执行什么
- **观测增强已固化**：z 参考点标注（tcp=指尖中点 / 物体=center）、`holding: yes (物体名)`、`history:` 抓取/释放事件日志、`contact:` 机械臂×任务物体接触力（SAPIEN get_contacts，F/T 传感器类比）、逐步计时 `t_brain_s/t_env_s`
- **实测时延结构**：brain 中位 ~23s/步（max 268s，网关波动），env 0.062s——决策频率 ≈0.04Hz，brain:env = 380:1
- **对照组（纯记忆模块，60 步）**：s14 抓住，s16 甩脱一次后连续 43 步稳定持有（v2 振荡已消失），但贴地（z≈0.04-0.05）徘徊，未尝试对准绿块，60 步耗尽
- **力觉轮（+contact 行，60 步）**：携带高度抬到 0.089（对照 0.049），s46 对准绿块下降，s47 cubeB 接触力 224N（压上绿块顶），**s52 在绿块正上方释放，红块物理堆叠成功并保持到终局（red_z 0.060 稳定 8 步）**——但 env success 未触发、模型不知道"已完成"，随后 8 步复抓/推搡毁掉判定机会（s55 单步思考峰值 27k 字）
- 距首次成功差的是"**任务状态可见性**"：env 的 success 判定结果不在 state_text 里，模型无从知道"已经堆好了，退开即可"。候选：state_text 加 `task status: in progress/complete`（env 判据回灌，Inner-Monologue 式任务监视器；是否违反无特权信息原则待定夺）
- **success 未触发的真正原因已查明**（读 env 源码）：success = on_cubeB × static × **(~is_grasping)**——力觉轮 s52 位置/静止条件全过，但模型"松爪"只松到 0.45 且指尖仍搭在红块上，is_grasping 判 True。已删除帮倒忙的 "never while waiting" 规则（90d8f7c）
- **夹爪状态三态化**（90d8f7c + 85bd725 通用化）：`gripper: grasped <物体> / touching <物体> (<力>N) / free (no contact)`，接触检测任务无关（按关节链组件识别机器人，任何非机器人刚体都报，含桌面反力）
- **三态验证轮（60 步，未成功）**：信号层完美（grasped/touching/free 转换全部准确及时，s40 中途松手立即显形 touching+力峰，s44 模型自发纠错重抓），但行为层随机性主导——s40 无预兆中途松手浪费 8 步，60 步耗尽在横移爬行段，未到释放。**结论：链路各环节均已被单独验证（接近/抓取/携带/对准/释放/堆叠成型 ✓），单集成败已被决策随机性主导，单集调试收益递减**——下一步应在"多集统计"或"episode state 子目标锚点（M1）"之间做选择

**2026-10-16 协议 v2（StackCube 首次通关，`runs/v2_skills_stackcube/`）**：用户裁定将大脑输出改为**技能调用**——`{"dx,dy,dz(底盘移动,固定臂恒0),skill:"none|pick|place","object":"<名>"}`；抓取/放置由 adapter 内的闭环原语执行（20Hz 伺服 + 力反馈，含全部手臂运动；dx/dy/dz 语义改为移动底盘，跨本体通用，固定臂忽略）。大脑职责收敛为：选技能、选对象、选时机。**首跑即通关：2 次大脑调用（pick cube_red → place cube_green）、21 仿真步、3.5k token、墙钟 23.5s——对比 v1 架构 30-60 次调用、15-47 分钟、全败**。接口冻结条款按其预留的版本变更通道执行（用户批准）。技能注册表天然支持扩展（push/pull 等后续按任务族添加）。

**Run 纪律（教训）**：启动新 run 前 `pgrep -af "harness run"` 确认旧的已退；同一 `--out` 目录绝不并发两个 run——frames 按迭代号同名覆盖、episodes JSON 后写者独占，图像产物污染不可恢复（2026-10-16 曾发生：首个 v2 run 未确认死亡即重发，成功数值幸存但帧/视频被污染，重跑净化）。另：SAPIEN 进程偶发退出挂起（工作已完成、进程不自灭），直接 kill 无损。

## 6. 下一步（按优先级）

2026-10-15 已完成原 1-4 项：

1. **last-known-position 记忆**（已做）：adapter 持有 `_last_seen`，不可见时状态文本给 "last seen N step(s) ago at ..." + delta/dist；`ObjectReading.last_seen_pos/last_seen_age`，落盘进 perception JSON
2. **首轮解析失败原因落盘**（已做）：`BrainOutput.parse_failures`（stage/error/完整原话），轨迹 JSON + 逐集页面均展示
3. **腕部相机俯角 120°→135°**（已做）：近距目检方块+指间全程在帧内，接近段不再失明
4. **预算示例提额**（已做）：250 calls / 1.5M tokens（§4 命令已更新）

后备/中期：近距若再现失明可降 WRIST_MIN_PIXELS（200→50）；LIBERO 适配器（envs 接口已预留）；分割换 MobileSAM/YOLO-World（`perceive()` 接口不变，可解决 PegInsertionSide）；方案 B（模型主动调用测量工具）备选

## 7. Git / 发布约定

- 本地 `main` 保留完整历史；`v0.9-legacy` tag 封存旧项目；旧数据在 `~/embodied-archive-v0.9/`
- 远端 `magickkdd/general-robo-harness` 只推**当前快照**（两边历史不同，不能直接 push）：

```bash
git checkout --orphan ship && git add -A && git commit -m "snapshot" \
  && git push -f https://<token>@github.com/magickkdd/general-robo-harness.git ship:main \
  && git checkout main && git branch -D ship
```

- 直连 GitHub push 常超时（408/52），重试即可通过；**不要**把 token 走第三方代理域名
- 评测产物不进 git（runs/ reports/ 已 ignore）；对话中暴露过的 API key / GitHub PAT 建议轮换（`.env` 改一处即可）

## 8. 设计原则（新贡献者必读）

- **接口冻结（2026-10-15 起）**：大脑 I/O 协议以 `harness/envs/base.py` 为准——输入 = 图像（`--images` 控制路数）+ state_text，输出 = 严格 JSON `[dx, dy, dz, grip] ∈ [-1,1]`。**harness 不得修改、重解释或放大模型的动作输出**（auto_cruise 教训：模型输出什么就执行什么）；协议变更必须显式升版本并全量重跑基线
- **极简**：非必要不加；用完不用的东西及时删。当前 ~1250 行，约束 ≤2000
- **无特权信息**：进模型输入的一切（图像、深度、坐标）必须来自真实机器人可获得的传感器；特权真值只用于离线验收
- **诚实评测**：success 采用环境判据；"not visible" 是正确输出而非失败；预算封顶 + 成功即停
- **每步可审计**：模型看到什么、想了什么、输出了什么、对错与否，全部落盘可回放
