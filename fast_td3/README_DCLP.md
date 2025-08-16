# DCLP Training System for IsaacLab Environments

这是一个基于FastTD3的分布式分类学习策略(DCLP)训练系统，专门为IsaacLab导航环境设计。

## 系统架构

### 核心组件

1. **DCLP模型** (`dclp.py`)
   - `MLPGaussianPolicy`: 高斯混合模型策略网络
   - `MLPActorCritic`: Actor-Critic架构
   - `DCLP`: 完整的训练算法类

2. **训练脚本** (`train_dclp_isaac_simple.py`)
   - 简化的DCLP训练流程
   - 支持模拟环境和真实IsaacLab环境
   - 经验回放和模型保存功能

3. **工具函数** (`dclp_utils.py`)
   - CNN网络用于LiDAR数据处理
   - 数学工具函数(激活函数、概率计算等)
   - MLP网络实现

## 特性

### LiDAR数据处理
- 支持270维LiDAR输入(90个点，每个点3个特征)
- 使用1D CNN进行特征提取
- 自适应激活函数用于距离信息处理

### 分布式策略学习
- 高斯混合模型(GMM)策略
- 4个混合组件的分布式动作采样
- 熵正则化的Actor-Critic学习

### 高效训练
- 经验回放缓冲区
- 软目标网络更新
- 双Q网络减少过估计

## 使用方法

### 1. 基本训练

```bash
cd /home/unnc/FastTD3/fast_td3

# 使用模拟环境进行测试
python train_dclp_isaac_simple.py --total_timesteps 10000 --device cpu

# 使用GPU加速训练
python train_dclp_isaac_simple.py --total_timesteps 1000000 --device cuda
```

### 2. 参数配置

```bash
# 自定义学习率和批次大小
python train_dclp_isaac_simple.py \
    --learning_rate 1e-4 \
    --batch_size 512 \
    --buffer_size 2000000 \
    --total_timesteps 2000000

# 调整训练频率和保存频率
python train_dclp_isaac_simple.py \
    --train_freq 4 \
    --save_freq 25000 \
    --eval_freq 5000
```

### 3. 使用真实IsaacLab环境

确保正确安装IsaacLab后：

```bash
# 使用Turtlebot2导航环境
python train_dclp_isaac_simple.py \
    --env_name Isaac-Navigation-Flat-Turtlebot2-v0 \
    --total_timesteps 1000000
```

## 测试系统

### 运行模块测试

```bash
python test_dclp.py
```

这将测试：
- PyTorch基础功能
- DCLP模块导入
- 网络实例化
- 前向传播

## 文件结构

```
fast_td3/
├── dclp.py                     # DCLP算法核心实现
├── dclp_utils.py              # 工具函数和网络定义
├── train_dclp_isaac_simple.py # 简化训练脚本
├── test_dclp.py               # 模块测试脚本
├── fast_td3_utils.py          # FastTD3工具函数
└── environments/
    └── isaaclab_env.py        # IsaacLab环境包装器
```

## 模型保存和加载

### 训练过程中的保存
- 每隔50,000步自动保存模型
- 训练结束时保存最终模型
- 保存路径：`dclp_model_<timestep>.pt`或`dclp_model_final.pt`

### 手动加载模型

```python
from dclp import DCLP

# 创建DCLP实例
dclp = DCLP(state_dim=280, action_dim=2, device='cuda')

# 加载训练好的模型
dclp.load('dclp_model_final.pt')

# 使用模型进行推理
state = env.reset()
action = dclp.get_action(state, deterministic=True)
```

## 性能优化建议

### 训练效率
1. **使用GPU**: 设置`--device cuda`加速训练
2. **调整批次大小**: 根据GPU内存调整`--batch_size`
3. **经验回放**: 增加`--buffer_size`提高样本效率

### 超参数调优
1. **学习率**: 从3e-4开始，根据收敛情况调整
2. **训练频率**: 对于复杂环境可以降低`--train_freq`
3. **熵系数**: 在DCLP类中调整`alpha`参数控制探索

### 环境兼容性
- 确保LiDAR输入维度为270(90个点×3个特征)
- 动作空间应为2维连续动作
- 支持Gymnasium接口的环境

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖模块在正确路径
2. **维度不匹配**: 检查环境的观察和动作空间维度
3. **设备错误**: 确保模型和数据在同一设备上
4. **内存不足**: 减少批次大小或缓冲区大小

### 调试技巧

```bash
# 启用详细日志
python -u train_dclp_isaac_simple.py --total_timesteps 1000

# 使用CPU避免GPU问题
python train_dclp_isaac_simple.py --device cpu

# 小规模测试
python train_dclp_isaac_simple.py --total_timesteps 100 --learning_starts 50
```

## 实验结果

训练成功完成后，你应该看到：
- 平均奖励逐渐增长
- Actor和Critic损失趋于稳定
- 评估性能持续改善
- 模型文件成功保存

使用模拟环境的快速测试表明系统运行正常，可以开始实际的强化学习训练。
