# harness

通用跨本体、跨 VLM 大脑的具身评测 harness（ManiSkill3 + OpenAI 兼容 API）。

- **环境轴**：ManiSkill3，任务 × 本体矩阵（panda / xarm6_robotiq / fetch），统一 EE 增量动作 `[dx, dy, dz, grip]`
- **大脑轴**：任意 OpenAI 兼容 API（`--base-url --model --api-key-env`），图像 + 状态文本观测，输出动作 JSON
- **评测**：成功率 / 步数 / token 花费，预算封顶，JSON + HTML 报告

## 安装

```bash
uv venv --python /usr/bin/python3.10 .venv
uv pip install --python .venv/bin/python torch==2.14.1+cpu \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/
uv pip install --python .venv/bin/python -e ".[sim,dev]" \
  --index-url https://mirrors.aliyun.com/pypi/simple/
```

（无 NVIDIA Vulkan ICD 的机器自动走 lavapipe 软渲染；机器人资产从 hf-mirror 下载到 `~/.maniskill/`）

## 使用

```bash
harness list                                   # 任务 × 本体矩阵
harness run --task PickCube-v1 --embodiment panda \
  --brain random --episodes 2                  # 零成本全链路冒烟
harness run --task PickCube-v1 --embodiment panda \
  --brain vlm --model <vision-model> --base-url <endpoint> \
  --api-key-env LLM_API_KEY --episodes 3       # 真实 VLM 评测
harness run --all-cells --brain vlm ...        # 全矩阵批量
```

产物在 `runs/<时间戳>_<...>/`：`episodes/*.json`（逐步轨迹）、`frames/`（可选抽帧）、`summary.json`、`report.html`。

测试：`.venv/bin/python -m pytest tests/ -q`（slow 标记为真实仿真集成测试）

旧版本（v0.9 及之前）见 git tag `v0.9-legacy`；数据归档于 `~/embodied-archive-v0.9/`。
