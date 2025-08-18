# DCLP Model Player Scripts

这里提供了两个脚本来播放和可视化训练好的 DCLP 模型：

## 1. 简单播放脚本 (`simple_play_dclp.py`)

这是一个简化的脚本，类似于 `simple_play.py`，用于快速测试 DCLP 模型。

### 用法
```bash
python simple_play_dclp.py <model_path> <task_name> [num_episodes]
```

### 示例
```bash
# 播放 3 个回合（默认）
python simple_play_dclp.py models/Isaac-Navigation-Flat-Turtlebot2-v0__DCLP__1_final.pt Isaac-Navigation-Flat-Turtlebot2-v0

# 播放 5 个回合
python simple_play_dclp.py models/Isaac-Navigation-Flat-Turtlebot2-v0__DCLP__1_final.pt Isaac-Navigation-Flat-Turtlebot2-v0 5
```

## 2. 完整播放脚本 (`play_dclp_isaac_lab.py`)

这是一个功能完整的脚本，提供更多的配置选项和详细的统计信息。

### 用法
```bash
python play_dclp_isaac_lab.py --model_path <path> --task_name <task> [options]
```

### 参数说明
- `--model_path`: DCLP 模型检查点文件路径（必需）
- `--task_name`: IsaacLab 任务名称（必需）
- `--device`: 运行设备（cuda/cpu，默认：cuda）
- `--num_episodes`: 播放回合数（默认：5）
- `--max_episode_steps`: 每回合最大步数（默认：使用环境默认值）
- `--deterministic`: 使用确定性策略（无探索噪声）
- `--seed`: 随机种子（默认：42）
- `--action_bounds`: 动作边界（默认：1.0）
- `--fps`: 目标帧率（默认：60）
- `--headless`: 无头模式运行（不显示GUI）

### 示例
```bash
# 基本用法
python play_dclp_isaac_lab.py --model_path models/model.pt --task_name Isaac-Navigation-Flat-Turtlebot2-v0

# 播放 10 个回合，使用确定性策略
python play_dclp_isaac_lab.py --model_path models/model.pt --task_name Isaac-Navigation-Flat-Turtlebot2-v0 --num_episodes 10 --deterministic

# 在 CPU 上运行，30 FPS
python play_dclp_isaac_lab.py --model_path models/model.pt --task_name Isaac-Navigation-Flat-Turtlebot2-v0 --device cpu --fps 30

# 无头模式运行（不显示GUI）
python play_dclp_isaac_lab.py --model_path models/model.pt --task_name Isaac-Navigation-Flat-Turtlebot2-v0 --headless
```

## 支持的任务

这些脚本主要设计用于 IsaacLab 导航任务，例如：
- `Isaac-Navigation-Flat-Turtlebot2-v0`
- `Isaac-Navigation-Rough-Turtlebot2-v0`
- 其他兼容的导航任务

## 模型格式要求

脚本支持两种 DCLP 模型格式：

1. **DCLP 原生格式**：包含 `actor_critic` 键的检查点
2. **训练检查点格式**：包含 `args`、`obs_normalizer_state` 等键的完整训练状态

## 故障排除

### 常见问题

1. **维度不匹配**
   - 确保模型是为当前任务训练的
   - 检查观测空间和动作空间维度是否匹配

2. **CUDA 内存不足**
   - 使用 `--device cpu` 在 CPU 上运行
   - 或减少 `--num_episodes`

3. **GUI 无法显示**
   - 确保有适当的显示环境（X11/Wayland）
   - 或使用 `--headless` 模式

4. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保模型文件格式正确
   - 检查依赖包是否完整安装

### 调试信息

脚本会显示详细的加载和运行信息，包括：
- 模型参数（状态空间、动作空间维度）
- 设备信息
- 环境配置
- 每回合的统计信息

## 注意事项

1. **IsaacSim 环境**：确保 IsaacSim 正确安装并配置
2. **GPU 驱动**：如果使用 CUDA，确保 GPU 驱动和 CUDA 版本兼容
3. **Python 环境**：确保所有必要的包都已安装（torch、isaaclab 等）
4. **模型兼容性**：确保模型是用兼容的 DCLP 版本训练的
