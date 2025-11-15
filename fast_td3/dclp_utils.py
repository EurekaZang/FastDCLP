import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def count_vars(pytorch_model):
    """
    统计给定 PyTorch 模型中所有可训练参数的总数。

    参数：
        pytorch_model (torch.nn.Module): 需要统计参数数量的 PyTorch 模型。

    返回：
        int: 模型中所有可训练参数的总数。
    """

    return sum(param.numel() for param in pytorch_model.parameters() if param.requires_grad)

def clip_with_gradient_passthrough(input_tensor, lower_bound=-1., upper_bound=1.):
    """
    在前向传播时将输入张量的值裁剪到指定的上下界，
    但在反向传播时允许梯度像未裁剪一样通过。
    参数：
        input_tensor (torch.Tensor): 需要裁剪的输入张量。
        lower_bound (float, 可选): 裁剪的下界，默认-1.0。
        upper_bound (float, 可选): 裁剪的上界，默认1.0。
    返回：
        torch.Tensor: 前向传播时被裁剪到指定范围的张量，
                      但梯度不受裁剪影响。
    """

    clip_upper_mask = (input_tensor > upper_bound).float()
    clip_lower_mask = (input_tensor < lower_bound).float()
    return input_tensor + (upper_bound - input_tensor) * clip_upper_mask + (lower_bound - input_tensor) * clip_lower_mask

def clip_min_with_gradient_passthrough(input_tensor, lower_bound=EPS):
    """
    将输入张量中小于指定下界的值裁剪到下界，
    但在反向传播时允许梯度像未裁剪一样通过。

    该函数在前向传播时将所有小于 lower_bound 的元素设为 lower_bound，
    但在反向传播时保留原始梯度。适用于推理时需要最小值约束但不希望影响梯度计算的场景。

    参数：
        input_tensor (torch.Tensor): 需要裁剪的输入张量。
        lower_bound (float, 可选): 最小裁剪值，默认为 EPS。

    返回：
        torch.Tensor: 小于 lower_bound 的值被裁剪，但梯度不受影响的张量。
    """

    clip_low_mask = (input_tensor < lower_bound).float()
    lower_bound_tensor = torch.tensor(lower_bound, device=input_tensor.device, dtype=input_tensor.dtype)
    return input_tensor + (lower_bound_tensor - input_tensor) * clip_low_mask

def reciprocal_relu(input_features, alpha_activation):
    """
    对输入张量应用类似于 ReLU 的倒数激活函数。

    该函数先对输入加上 alpha_activation 参数后，
    用自定义裁剪函数（clip_min_with_gradient_passthrough，最小值为 EPS）处理，
    然后取倒数。梯度在裁剪操作中被保留。
    参数：
        input_features (torch.Tensor): 需要激活的输入张量。
        alpha_activation (float 或 torch.Tensor): 激活前加到输入上的 alpha 参数。
    返回：
        torch.Tensor: 应用激活函数后的结果。
    """

    # 检查输入是否包含 NaN 或 inf
    if torch.any(torch.isnan(input_features)) or torch.any(torch.isinf(input_features)):
        input_features = torch.clamp(input_features, min=-10.0, max=10.0)
    
    clipped_input = clip_min_with_gradient_passthrough(input_features + alpha_activation, lower_bound=EPS)
    reciprocal_result = torch.reciprocal(clipped_input)
    
    # 检查输出是否包含 NaN 或 inf
    if torch.any(torch.isnan(reciprocal_result)) or torch.any(torch.isinf(reciprocal_result)):
        reciprocal_result = torch.clamp(reciprocal_result, min=EPS, max=1.0/EPS)
    
    return reciprocal_result

def create_log_gaussian(mean_tensor, log_std_tensor, sample_points):
    """
    计算样本在多元对角高斯分布下的对数概率。
    参数：
        mean_tensor (torch.Tensor): 高斯分布的均值，形状 (..., D)
        log_std_tensor (torch.Tensor): 高斯分布的对数标准差，形状 (..., D)
        sample_points (torch.Tensor): 需要计算对数概率的样本点，形状 (..., D)
    返回：
        torch.Tensor: 每个样本点在指定高斯分布下的对数概率，形状 (...,)
    说明：
        - 假设协方差矩阵为对角阵（各维独立）。
        - 使用对数标准差以保证数值稳定性。
    """

    # 计算标准化距离
    normalized_distance = (sample_points - mean_tensor) * torch.exp(-log_std_tensor)
    
    # 计算二次项
    quadratic_term = -0.5 * torch.sum(normalized_distance ** 2, dim=-1)

    # 计算归一化常数
    log_normalization = torch.sum(log_std_tensor, dim=-1)
    feature_dimension = float(mean_tensor.shape[-1])
    log_normalization += 0.5 * feature_dimension * np.log(2 * np.pi)

    log_probability = quadratic_term - log_normalization
    return log_probability

def apply_squashing_func(action_mean, sampled_action, log_action_prob):
    """
    对输入张量应用压缩函数（tanh），并对动作的对数概率进行雅可比修正。
    参数：
        action_mean (torch.Tensor): 需要压缩的均值张量。
        sampled_action (torch.Tensor): 需要压缩的采样动作张量。
        log_action_prob (torch.Tensor): 压缩前的动作对数概率。
    返回：
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - 压缩后的均值张量（action_mean，tanh 后）。
            - 压缩后的动作张量（sampled_action，tanh 后）。
            - 经雅可比修正后的对数概率（log_action_prob）。
    """
    # 应用tanh压缩
    squashed_mean = torch.tanh(action_mean)
    squashed_action = torch.tanh(sampled_action)

    # 雅可比校正：log|det(∂tanh(u)/∂u)| = log(1 - tanh²(u))
    jacobian_correction = torch.sum(torch.log(clip_with_gradient_passthrough(1 - squashed_action**2, lower_bound=0, upper_bound=1) + 1e-6), dim=1)
    adjusted_log_prob = log_action_prob - jacobian_correction

    return squashed_mean, squashed_action, adjusted_log_prob


"""
---------------------------------------------------------------------------------
                                Network Definitions
---------------------------------------------------------------------------------
"""

class CNNDense(nn.Module):
    """
    CNNDense is a PyTorch neural network module designed to process LiDAR and robot state data using 1D convolutional layers.
    Args:
        activation (callable, optional): Activation function to use after each convolutional layer. Defaults to F.leaky_relu.
        output_activation (callable, optional): Activation function to apply to the output. Defaults to None.
    Attributes:
        alpha_activation_param (torch.nn.Parameter): Learnable parameter for the custom activation applied to the LiDAR distance feature.
        conv_layer_1 (torch.nn.Conv1d): First 1D convolutional layer (input channels: 6, output channels: 32).
        conv_layer_2 (torch.nn.Conv1d): Second 1D convolutional layer (input channels: 32, output channels: 64).
        conv_layer_3 (torch.nn.Conv1d): Third 1D convolutional layer (input channels: 64, output channels: 128).
        max_pool (torch.nn.MaxPool1d): Global max pooling layer with kernel size 90.
        activation_func (callable): Activation function used after each convolutional layer.
    Forward Input:
        input_data (torch.Tensor): Input tensor of shape [batch_size, input_dim], where the first 6*90 elements represent LiDAR data
            (90 directions, 6 features per direction), and the remaining elements represent additional robot state information.
    Forward Output:
        torch.Tensor: Output tensor of shape [batch_size, output_dim], which is the concatenation of CNN-extracted features
            from LiDAR data and the remaining robot state information.
    """

    def __init__(self, activation=F.leaky_relu, output_activation=None):
        super(CNNDense, self).__init__()
        # 定义可训练激活参数
        self.alpha_activation_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.conv_layer_1 = nn.Conv1d(6, 32, kernel_size=1, stride=1, padding=0)
        self.conv_layer_2 = nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0)
        self.conv_layer_3 = nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool1d(kernel_size=90, stride=1, padding=0)
        self.activation_func = activation

    def forward(self, input_data):
        """
        Performs a forward pass through the neural network module for processing LiDAR and robot state data.

        Args:
            input_data (torch.Tensor): Input tensor of shape [batch_size, input_dim], where the first 6*90 elements
                correspond to LiDAR data (90 directions, 6 features per direction), and the remaining elements
                represent additional robot state information.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim], which is the concatenation of
                extracted CNN features from LiDAR data and the remaining robot state information.

        Processing Steps:
            1. Extracts and reshapes the LiDAR data segment from the input.
            2. Applies a custom activation function to the distance feature of the LiDAR data.
            3. Recombines the processed distance feature with other LiDAR features.
            4. Transposes the data for Conv1d compatibility.
            5. Passes the data through three 1D convolutional layers with LeakyReLU activations.
            6. Applies global max pooling and flattens the resulting features.
            7. Concatenates the CNN-extracted features with the remaining robot state information.
        """
        # 提取和重塑激光雷达数据
        lidar_data = input_data[:, 0:6*90]  # 激光雷达数据段
        reshaped_lidar = lidar_data.view(-1, 90, 6)  # [batch, 90方向, 6特征]
        # 对距离特征应用自定义激活
        processed_distance = reciprocal_relu(reshaped_lidar[:, :, 2], self.alpha_activation_param)  # 处理距离信息(第3个特征)
        # 重新组合特征
        combined_lidar_features = torch.cat([
            reshaped_lidar[:, :, 0:2],          # 方向信息(cos, sin)
            processed_distance.unsqueeze(-1),   # 处理后的距离
            reshaped_lidar[:, :, 3:6]           # 机器人几何信息(length1, length2, width)
        ], dim=-1)
        # 转换为Conv1d格式: [batch, features, sequence]
        transposed_features = combined_lidar_features.transpose(1, 2)
        # 三层1D卷积提取特征
        conv1_output = F.leaky_relu(self.conv_layer_1(transposed_features))
        conv2_output = F.leaky_relu(self.conv_layer_2(conv1_output))
        conv3_output = F.leaky_relu(self.conv_layer_3(conv2_output))
        # 全局最大池化
        pooled_features = self.max_pool(conv3_output)
        flattened_cnn_features = pooled_features.view(pooled_features.size(0), -1)
        # 拼接CNN特征和其他状态信息
        return torch.cat([flattened_cnn_features, input_data[:, 6*90:]], dim=-1)

class CNNNet(nn.Module):
    """
    CNNNet is a 1D Convolutional Neural Network module for feature extraction from sequential data.
    Args:
        activation (callable, optional): Activation function to use after each convolutional layer. Default is F.relu.
        output_activation (callable, optional): Activation function to use at the output layer. Default is None.
    Layers:
        conv_layer_1 (nn.Conv1d): First 1D convolutional layer (in_channels=6, out_channels=32, kernel_size=1).
        conv_layer_2 (nn.Conv1d): Second 1D convolutional layer (in_channels=32, out_channels=64, kernel_size=1).
        conv_layer_3 (nn.Conv1d): Third 1D convolutional layer (in_channels=64, out_channels=128, kernel_size=1).
        max_pool (nn.MaxPool1d): 1D max pooling layer (kernel_size=90, stride=1).
    Forward Input:
        input_tensor (torch.Tensor): Input tensor of shape [batch_size, sequence_length, features].
        additional_input (optional): Unused, for compatibility.
    Forward Output:
        torch.Tensor: Flattened feature tensor after convolution and pooling, shape [batch_size, features_out].
    """

    def __init__(self, activation=F.relu, output_activation=None):
        super(CNNNet, self).__init__()
        self.conv_layer_1 = nn.Conv1d(6, 32, kernel_size=1, stride=1, padding=0)
        self.conv_layer_2 = nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0)
        self.conv_layer_3 = nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool1d(kernel_size=90, stride=1, padding=0)
        self.activation_func = activation
    
    def forward(self, input_tensor, additional_input=None):
        # input_tensor shape: [batch_size, sequence_length, features] -> [batch_size, features, sequence_length]
        transposed_input = input_tensor.transpose(1, 2)
        
        conv1_output = F.leaky_relu(self.conv_layer_1(transposed_input))
        conv2_output = F.leaky_relu(self.conv_layer_2(conv1_output))
        conv3_output = F.leaky_relu(self.conv_layer_3(conv2_output))
        pooled_output = self.max_pool(conv3_output)
        flattened_features = pooled_output.view(pooled_output.size(0), -1)
        return flattened_features

class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) implementation using PyTorch.

    Args:
        input_dim (int): The number of input features.
        hidden_sizes (list or tuple of int): Sizes of the hidden layers.
        activation (callable, optional): Activation function to use for hidden layers. Defaults to F.leaky_relu.
        output_activation (callable, optional): Activation function to use for the output layer. Defaults to None.
    Attributes:
        hidden_activation (callable): Activation function for hidden layers.
        output_activation (callable): Activation function for the output layer.
        layer_modules (nn.ModuleList): List of linear layers in the network.
    Methods:
        forward(input_tensor):
            Performs a forward pass through the network.
            Args:
                input_tensor (torch.Tensor): Input tensor to the network.
            Returns:
                torch.Tensor: Output tensor after passing through the MLP.
    """

    def __init__(self, input_dim, hidden_sizes, activation=F.leaky_relu, output_activation=None):
        super(MLP, self).__init__()
        self.hidden_activation = activation
        self.output_activation = output_activation
        layer_list = []
        layer_dimensions = [input_dim] + list(hidden_sizes)
        for layer_idx in range(len(layer_dimensions) - 1):
            layer_list.append(nn.Linear(layer_dimensions[layer_idx], layer_dimensions[layer_idx+1]))
        self.layer_modules = nn.ModuleList(layer_list)

    def forward(self, input_tensor):
        output_tensor = input_tensor
        for layer_idx, layer in enumerate(self.layer_modules):
            output_tensor = layer(output_tensor)
            if layer_idx < len(self.layer_modules) - 1:  # 隐藏层
                output_tensor = self.hidden_activation(output_tensor)
            else:  # 输出层
                if self.output_activation is not None:
                    output_tensor = self.output_activation(output_tensor)
        return output_tensor

class DistributionalQNetwork(MLP):
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        hidden_sizes: list,
        activation=F.leaky_relu,
        output_activation=None,
        num_atoms: int = 251,
        v_min: float = -100.0,
        v_max: float = 100.0,
    ):
        super().__init__(input_dim, hidden_sizes, activation, output_activation)
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        l_mask = torch.logical_and((u > 0), (l == u))
        u_mask = torch.logical_and((l < (self.num_atoms - 1)), (l == u))

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        return proj_dist

def calculate_turtlebot_velocities(omega_left_rad_s, omega_right_rad_s):
    """
    根据左右轮角速度计算TurtleBot2的线速度和角速度。

    Args:
        omega_left_rad_s (float): 左轮角速度 (弧度/秒).
        omega_right_rad_s (float): 右轮角速度 (弧度/秒).

    Returns:
        tuple: 包含线速度 (米/秒) 和角速度 (弧度/秒) 的元组.
    """
    # TurtleBot2 的尺寸参数 (使用您提供的数据)
    wheel_diameter_mm = 72.0
    wheelbase_mm = 235.0

    # 将单位转换为米
    wheel_radius_m = (wheel_diameter_mm / 2.0) / 1000.0  # 72mm / 2 = 36mm = 0.036 m
    wheelbase_m = wheelbase_mm / 1000.0             # 235mm = 0.235 m

    # 计算 TurtleBot2 的线速度
    linear_velocity_m_s = wheel_radius_m * (omega_left_rad_s + omega_right_rad_s) / 2.0

    # 计算 TurtleBot2 的角速度
    # 注意：这里右轮减左轮是通常的约定，会产生正角速度（逆时针旋转）
    angular_velocity_rad_s = wheel_radius_m * (omega_right_rad_s - omega_left_rad_s) / wheelbase_m

    return linear_velocity_m_s, angular_velocity_rad_s

@dataclass
class DCLPArgs:
    """DCLP training arguments"""
    # Environment
    env_name: str = "Isaac-Navigation-Flat-Turtlebot2-v0"
    """IsaacLab environment name"""

    #GUI / rendering
    headless: bool = True
    render_mode: str = "human"

    # Training
    seed: int = 42
    """Random seed"""
    total_timesteps: int = 2000000
    """Total training timesteps"""
    learning_starts: int = 25000
    """Steps before learning starts"""
    num_envs: int = 1024
    """Number of parallel environments"""

    # Algorithm
    agent: str = "dclp"
    """Agent type"""
    gamma: float = 0.99
    """Discount factor"""
    num_steps: int = 8
    """the number of steps to use for the multi-step return"""
    tau: float = 0.005
    """Target network soft update rate"""
    batch_size: int = 4096
    """Training batch size"""
    buffer_size: int = 1024 * 5
    """Replay buffer size"""
    alpha: float = 0.01
    """Entropy regularization coefficient"""

    # Learning rates
    actor_learning_rate: float = 1e-4
    """Actor learning rate"""
    critic_learning_rate: float = 1e-4
    """Critic learning rate"""
    actor_learning_rate_end: float = 1e-5
    """Actor final learning rate"""
    critic_learning_rate_end: float = 1e-5
    """Critic final learning rate"""

    # Network architecture
    narrow_layers: bool = True
    """Use narrow layers as FastTD3"""
    num_hidden_layers: int = 4
    """Number of hidden layers"""
    actor_hidden_dim: int = 128
    """Actor hidden dimension"""
    critic_hidden_dim: int = 128
    """Critic hidden dimension"""
    init_scale: float = 0.01
    """Actor initialization scale"""

    # TD3 specific
    policy_noise: float = 0.1
    """Policy noise for target smoothing"""
    noise_clip: float = 0.5
    """Noise clipping for target smoothing"""
    policy_frequency: int = 2
    """Policy update frequency"""
    num_updates: int = 1
    """Number of updates per step"""

    # Distributional Critic 
    num_atoms: int = 251
    """Number of atoms for distributional critic"""
    v_min: float = -100.0
    """Minimum value for distributional critic"""
    v_max: float = 100.0
    """Maximum value for distributional critic"""

    # Normalization
    obs_normalization: bool = False
    """Use observation normalization"""
    reward_normalization: bool = False
    """Use reward normalization"""

    # Hardware
    cuda: bool = True
    """Use CUDA if available"""
    device_rank: int = 0
    """Device rank"""
    torch_deterministic: bool = False
    """PyTorch deterministic mode"""

    # Optimization
    amp: bool = True
    """Use automatic mixed precision"""
    amp_dtype: str = "fp16"
    """AMP dtype (bf16 or fp16)"""
    compile: bool = True
    """Compile model with torch.compile"""
    compile_mode: str = "reduce-overhead"
    """Compile mode"""
    weight_decay: float = 0.0
    """Weight decay"""
    use_grad_norm_clipping: bool = True
    """Use gradient norm clipping"""
    max_grad_norm: float = 0.5
    """Maximum gradient norm"""

    # Logging
    use_wandb: bool = True
    """Use Weights & Biases logging"""
    project: str = "DCLP-IsaacLab"
    """W&B project name"""
    exp_name: str = "dclp_training"
    """Experiment name"""
    eval_interval: int = 100000
    """Evaluation interval"""
    save_interval: int = 100000
    """Model save interval"""

    # Environment specific
    action_bounds: Optional[float] = None
    """Action bounds scaling"""
    max_episode_steps: Optional[int] = None
    """Maximum episode steps override"""

    # DCLP specific
    lidar_points: int = 90
    """Number of LiDAR points (270/3)"""
    lidar_features: int = 3
    """LiDAR features per point (sin, cos, distance)"""
    use_cnn_features: bool = True
    """Use CNN for LiDAR feature extraction"""

    def __post_init__(self):
        # Sanitize render_mode to handle Windows cmd single quotes and curly quotes
        if isinstance(self.render_mode, str):
            rm = self.render_mode.strip()
            # Strip common quote characters and whitespace
            rm = rm.strip("'\"“”")
            if rm.lower() in ("none", ""):
                self.render_mode = None
            else:
                self.render_mode = rm