import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def count_vars(pytorch_model):
    """
    Counts the total number of trainable parameters in a given PyTorch model.

    Args:
        pytorch_model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """

    return sum(param.numel() for param in pytorch_model.parameters() if param.requires_grad)

def clip_with_gradient_passthrough(input_tensor, lower_bound=-1., upper_bound=1.):
    """
    Clips the values of the input tensor to the specified lower and upper bounds during the forward pass,
    but allows gradients to pass through as if no clipping was applied during the backward pass.
    Args:
        input_tensor (torch.Tensor): The input tensor to be clipped.
        lower_bound (float, optional): The lower bound to clip to. Default is -1.0.
        upper_bound (float, optional): The upper bound to clip to. Default is 1.0.
    Returns:
        torch.Tensor: The tensor with values clipped to the specified bounds in the forward pass,
                      but with gradients unaffected by the clipping operation.
    """

    clip_upper_mask = (input_tensor > upper_bound).float()
    clip_lower_mask = (input_tensor < lower_bound).float()
    return input_tensor + (upper_bound - input_tensor) * clip_upper_mask + (lower_bound - input_tensor) * clip_lower_mask

def clip_min_with_gradient_passthrough(input_tensor, lower_bound=EPS):
    """
    Clips the values of the input tensor below a specified lower bound, but allows gradients to pass through as if no clipping occurred.

    This function sets all elements of `input_tensor` that are less than `lower_bound` to `lower_bound` in the forward pass, 
    while preserving the original gradients during backpropagation. This is useful in scenarios where you want to enforce 
    a minimum value constraint during inference but do not want the clipping operation to affect gradient computation.

    Args:
        input_tensor (torch.Tensor): The input tensor to be clipped.
        lower_bound (float, optional): The minimum value to clip to. Defaults to EPS.

    Returns:
        torch.Tensor: The tensor with values clipped below `lower_bound`, but with gradients unaffected by the clipping.
    """

    clip_low_mask = (input_tensor < lower_bound).float()
    lower_bound_tensor = torch.tensor(lower_bound, device=input_tensor.device, dtype=input_tensor.dtype)
    return input_tensor + (lower_bound_tensor - input_tensor) * clip_low_mask

def reciprocal_relu(input_features, alpha_activation):
    """
    Applies a reciprocal ReLU-like activation function to the input tensor.

    This function computes the reciprocal of the input tensor added to an alpha activation parameter,
    after applying a custom clipping function (`clip_min_with_gradient_passthrough`) with a lower bound of EPS.
    The gradient is preserved through the clipping operation.
    Args:
        input_features (torch.Tensor): The input tensor to apply the activation function to.
        alpha_activation (float or torch.Tensor): The alpha parameter to be added to the input tensor before activation.
    Returns:
        torch.Tensor: The result of applying the modified activation function.
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
    Computes the log-probability of samples under a multivariate diagonal Gaussian distribution.
    Args:
        mean_tensor (torch.Tensor): The mean of the Gaussian distribution. Shape: (..., D)
        log_std_tensor (torch.Tensor): The log standard deviation of the Gaussian distribution. Shape: (..., D)
        sample_points (torch.Tensor): The points at which to evaluate the log-probability. Shape: (..., D)
    Returns:
        torch.Tensor: The log-probabilities of the sample points under the specified Gaussian. Shape: (...,)
    Notes:
        - Assumes a diagonal covariance matrix (i.e., independent dimensions).
        - The function uses the log standard deviation for numerical stability.
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
    Applies a squashing function (tanh) to the input tensors and adjusts the log-probabilities
    with the appropriate Jacobian correction.
    Args:
        action_mean (torch.Tensor): The mean tensor to be squashed.
        sampled_action (torch.Tensor): The sampled action tensor to be squashed.
        log_action_prob (torch.Tensor): The log-probabilities of the actions before squashing.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Squashed mean tensor (action_mean) after applying tanh.
            - Squashed action tensor (sampled_action) after applying tanh.
            - Adjusted log-probabilities (log_action_prob) with Jacobian correction for tanh.
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
        lidar_data = input_data[:, 0:3*90]                                                          # 激光雷达数据段
        reshaped_lidar = lidar_data.view(-1, 90, 3)                                                 # [batch, 90方向, 3特征]
        processed_distance = reciprocal_relu(reshaped_lidar[:, :, 2], self.alpha_activation_param)  # 处理距离信息(第3个特征)
        # 重新组合特征
        combined_lidar_features = torch.cat([
            reshaped_lidar[:, :, 0:2],                                                              # 方向信息(cos, sin)
            processed_distance.unsqueeze(-1),                                                       # 处理后的距离
        ], dim=-1)
        transposed_features = combined_lidar_features.transpose(1, 2)                               # 转换为Conv1d格式: [batch, features, sequence]

        conv1_output = F.leaky_relu(self.conv_layer_1(transposed_features))                         # 三层1D卷积提取特征
        conv2_output = F.leaky_relu(self.conv_layer_2(conv1_output))
        conv3_output = F.leaky_relu(self.conv_layer_3(conv2_output))

        pooled_features = self.max_pool(conv3_output)                                               # 全局最大池化
        flattened_cnn_features = pooled_features.view(pooled_features.size(0), -1)
        return torch.cat([flattened_cnn_features, input_data[:, 3*90:]], dim=-1)                    # 拼接CNN特征和其他状态信息

class CNNNet(nn.Module):
    """
    CNNNet is a 1D Convolutional Neural Network module for feature extraction from sequential data.
    Args:
        activation (callable, optional): Activation function to use after each convolutional layer. Default is F.relu.
        output_activation (callable, optional): Activation function to use at the output layer. Default is None.
    Layers:
        conv_layer_1 (nn.Conv1d): First 1D convolutional layer (in_channels=3, out_channels=32, kernel_size=1).
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
        self.conv_layer_1 = nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0)
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