# AIRC-ROS2-Istruction

AIRC-ROS2-Istruction 是一个面向移动机器人局部导航的 PPO 训练项目。当前训练环境使用 MuJoCo 表示围墙、随机箱形障碍、机器人和目标点，并用 2D LiDAR 射线构造策略观测。环境遵循 Gymnasium 的 `reset/step` 接口，也提供 PPO 使用的批量适配器。

![AIRC-ROS2-Istruction cover](assets/cover.jpg)

## 环境与策略

每个 episode 都会随机生成一张封闭场地地图、障碍物、起点、机器人朝向和目标点。地图与 LiDAR 查询使用 MuJoCo；机器人按差速运动学更新位姿，碰撞使用圆形机器人 footprint 与墙体、障碍物进行几何检查。碰撞和到达目标会结束 episode，超过最大步数会截断 episode。

这是适合讲解 PPO、观测、奖励和地图随机化的轻量二维场景。它没有模拟车轮动力学、惯性、打滑或传感器噪声；需要研究这些效应时，应继续扩展机器人模型和接触动力学。

`env/mujoco_env.py` 包含两个入口：

- `MujocoNavEnv` 是标准 Gymnasium 单环境，可用于示例程序、Notebook 和学生练习。
- `MujocoBatchEnv` 管理多个独立 MuJoCo 场景，并把观测、奖励、终止状态转换为 PPO 训练器使用的 PyTorch 张量。

环境在 CPU 上生成场景并计算射线；策略网络仍可在 CPU 或 CUDA 上运行。默认使用 8 个环境，实际训练速度取决于 CPU、策略网络和射线数量。

## 观测与动作

观测向量长度为 `R + 7`：

```text
[rays_norm(R),
 sin_goal, cos_goal,
 prev_vx_norm, prev_omega_norm,
 delta_vx_norm, delta_omega_norm,
 goal_distance_norm]
```

`R = ceil(2 * pi * patch_meters / ray_max_gap)`。默认 `patch_meters = 10.0`、`ray_max_gap = 0.6`，所以 `R = 105`、观测维度为 `112`。射线按机器人朝向排列，距离归一化到 `[0, 1]`。

动作是以物理单位表示的 `(vx, omega)`，分别为米/秒和弧度/秒。默认 `vx_forward_only = true`，所以线速度范围为 `[0, vx_max]`，角速度范围为 `[-omega_max, omega_max]`。策略内部的 tanh 高斯输出会映射到这些边界。

修改 `patch_meters` 或 `ray_max_gap` 会改变射线数和观测维度；修改动作边界会改变策略动作的物理含义。此类变更通常需要重新训练，并同步更新部署配置。

策略编码器由 `config/train_config.json` 的 `model` 字段选择，结构参数位于 `config/model_config.json`。当前默认 `circular_attn` 使用环形卷积与多查询注意力处理雷达射线；`cnn_zeropad`、`cnn_circular` 和 `mlp_2` 至 `mlp_5` 可用于比较不同编码方式。不同结构的 checkpoint 不能互换。

## 奖励和场景配置

环境配置位于 `config/env_config.json`：

| 部分 | 常用参数 | 含义 |
|---|---|---|
| `world` | `arena_size`、`obstacle_count`、`obstacle_min_size`、`obstacle_max_size` | 场地尺寸和随机障碍数量、尺寸 |
| `world` | `wall_thickness`、`robot_radius` | 围墙厚度和机器人 footprint 半径 |
| `obs` | `patch_meters`、`ray_max_gap` | LiDAR 量程和射线间距 |
| `sim` | `dt`、`max_episode_steps`、`task_point_success_radius_m` | 控制周期、最大步数和目标判定半径 |
| `limits` | `vx_max`、`omega_max`、`vx_forward_only` | 差速动作边界 |
| `reward` | `reward_progress`、`reward_collision`、`reward_success` | 目标进度、碰撞惩罚和到达奖励 |

奖励由目标距离变化、碰撞惩罚、到达目标奖励和动作平滑惩罚组成。固定相同随机种子可复现起始场景；不传种子时，每次 reset 都会重新采样地图和任务。

## 安装

先安装项目依赖，再按本机设备安装 PyTorch：

```bash
pip install -r requirements.txt

# CPU PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

MuJoCo Python 包、Gymnasium 和 TensorBoard 已列入 `requirements.txt`。运行交互窗口需要桌面图形环境；无桌面环境可以使用 `render_mode="rgb_array"` 做离屏渲染。导出 ONNX 还需要 `onnx` 和 `onnxruntime`。

## Gymnasium 示例

单环境使用标准 Gymnasium 返回值：`reset()` 返回 `(observation, info)`，`step(action)` 返回 `(observation, reward, terminated, truncated, info)`。

```python
import json
from env.mujoco_env import MujocoNavConfig, MujocoNavEnv

with open("config/env_config.json", encoding="utf-8") as f:
    config = MujocoNavConfig.from_dict(json.load(f))

env = MujocoNavEnv(config, render_mode="human")
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step([0.2, 0.0])
env.close()
```

## 课堂演示

启动随机地图和简单目标点控制器：

```bash
python -m tools.mujoco_demo --seed 0 --episodes 5
```

也可以用随机动作观察环境响应：

```bash
python -m tools.mujoco_demo --controller random --seed 0 --episodes 2
```

演示控制器不是学习策略。课堂上可以依次查看观测中的激光距离和目标方向，讨论碰撞惩罚、到达奖励，再启动 PPO 训练并比较训练前后的行为。

## PPO 训练

`config/train_config.json` 选择策略网络、并行环境数、rollout 长度、优化参数和总环境步数。默认设置使用 `circular_attn`、8 个环境和 200 万环境步。课堂试跑可以先减少 `run.total_env_steps`；增加障碍数量或射线数量会提高环境计算开销。

```bash
# 使用当前配置新建训练 run
python -m rl_ppo.train --fresh --tag mujoco

# 从 checkpoint 权重开始一组新实验
python -m rl_ppo.train --fresh --resume runs/<run>/latest.pt --tag finetune

# 在原 run 上恢复训练，包括优化器状态
python -m rl_ppo.train --resume runs/<run>
```

每次新训练会把训练配置、环境配置和模型配置复制到 `runs/<run>/`，便于复现实验。`--fresh --resume` 默认只载入策略权重；加 `--opt` 才同时恢复优化器状态。TensorBoard 默认使用端口 `6006`，可通过 `--port` 更改。

在主机桌面上实时查看 PPO 训练的第 0 个环境：

```bash
python3 -u -m rl_ppo.train --fresh --tag mujoco_visual --render --render-every 5 --render-fps 30
```

`--render-every` 指定每隔多少个批量环境步刷新窗口，`--render-fps 0` 取消显示限速。可加 `--total-env-steps 2048` 做短训练检查；省略则使用配置中的总步数。关闭窗口会保存 checkpoint 并停止训练。主机浏览器访问 `http://localhost:6006` 查看 TensorBoard 曲线。在 SSH 中启动时，需要把 `DISPLAY`、`XAUTHORITY` 设置为远程主机桌面会话的值。

## 评估与导出

加载 checkpoint 做纯可视化测试（冻结权重、确定性动作，默认 10 Hz；配置来自 checkpoint 所在 run）：

```bash
cd /home/pc/AIRC-ROS2-Istruction

python3 -u -m tools.mujoco_policy_test \
  --ckpt runs/20261118-195211-mujoco/step-258998272.pt \
  --seed 10000 \
  --episodes 64 \
  --fps 10
```

快速无窗口评估同一组场景：

```bash
cd /home/pc/AIRC-ROS2-Istruction

python3 -u -m tools.mujoco_policy_test \
  --ckpt runs/20261118-195211-mujoco/step-258998272.pt \
  --seed 10000 --episodes 64 --headless
```

`--headless` 可无窗口快速评估；`--fps 0` 可不限制显示速度。每个回合使用 `seed + 回合序号`，终端打印成功、碰撞或超时以及奖励。测试不会创建训练 run 或修改权重，关闭窗口或按 Ctrl+C 退出。

窄通道评估可以比较 DWA 与训练策略在固定地图上的表现：

```bash
python -m eval.scripts.narrow_channel_eval --ckpt runs/<run>/latest.pt --render
```

导出 checkpoint 为 ONNX：

```bash
python tools/export_onnx.py --ckpt runs/<run>/latest.pt --model circular_attn -o exported_model
```

导出目录包含 PyTorch 权重、ONNX 文件和描述动作边界等信息的 `meta.json`。改变编码器、观测维度或动作范围后，需使用匹配的配置加载权重。

## 代码结构

```text
AIRC-ROS2-Istruction/
├── config/                 # 训练、场景和模型配置
├── env/
│   ├── mujoco_env.py       # Gymnasium MuJoCo 场景及 PPO 批量适配器
│   ├── ray.py              # 射线工具（旧评估代码兼容）
│   └── utils.py            # 配置和日志工具
├── models/                 # PPO 策略与可选观测编码器
├── rl_ppo/                 # rollout、GAE、PPO 训练循环
├── eval/                   # DWA 和固定地图评估
├── tools/
│   ├── mujoco_demo.py      # 交互式场景演示
│   ├── mujoco_policy_test.py # 冻结策略的可视化测试和无窗口评估
│   └── export_onnx.py      # 模型导出
└── runs/                   # 训练日志和 checkpoint
```
