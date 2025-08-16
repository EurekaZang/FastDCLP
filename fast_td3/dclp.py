import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dclp_utils import *

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLPGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=[128, 128, 128, 128],
        activation=F.leaky_relu,
        output_activation=None
    ):
        super().__init__()
        self.num_mixture_components = 4
        self.hidden_size = [128, 128, 128, 128]
        self.net = MLP(
            input_dim=138, # the flattened_features has 128 elements, and the rest of the observation has 10 elements (280 - 270)
            hidden_sizes=self.hidden_size,
            activation=F.leaky_relu,
            output_activation=F.leaky_relu,
        )
        self.gmm_output_layer = nn.Linear(self.hidden_size[-1], (n_act * 2 + 1) * self.num_mixture_components)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # obs: [n_env * batch_size, n_obs] = [1024 * 32, 280] in each row the first 270 elements are LiDAR data
        # actions: [n_env * batch_size, n_act]
        batch_size = obs.shape[0]
        num_lidar_points = 90  # 270/3
        lidar_raw = obs[:, :270].reshape(batch_size, num_lidar_points, 3)
        lidar_raw[:, :, 2] = reciprocal_relu(lidar_raw[:, :, 2], self.alpha_activation_param)  # 处理距离信息(第3个特征)

        cnn = getattr(self, 'cnn', None)
        if cnn is None:
            self.cnn = CNNNet()
            cnn = self.cnn
        flattened_features = cnn(lidar_raw)
        obs_rest = obs[:, 270:]  # [batch_size, n_obs-270]
        obs_new = torch.cat([flattened_features, obs_rest], dim=1)
        x = torch.cat([obs_new, actions], 1)
        x = self.net(x)
        gmm_parameters = self.gmm_output_layer(x)
        reshaped_gmm_params = gmm_parameters.view(-1, self.num_mixture_components, 2*self.action_dimension+1)
        # divide parameters
        log_mixture_weights = reshaped_gmm_params[..., 0]              # [N, K]
        component_means = reshaped_gmm_params[..., 1:1+self.action_dimension]       # [N, K, action_dimension]
        log_component_std = reshaped_gmm_params[..., 1+self.action_dimension:]   # [N, K, action_dimension]
        # constrain log std to safe range
        constrained_log_std = torch.tanh(log_component_std)
        constrained_log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (constrained_log_std + 1)
        component_std_devs = torch.exp(constrained_log_std)
        selected_component_idx = torch.multinomial(torch.softmax(log_mixture_weights, dim=-1), num_samples=1)         # 选择组件

        batch_indices = torch.arange(batch_size, device=obs.device)                                                   # 获取选中组件的参数
        selected_component_mean = component_means[batch_indices, selected_component_idx.squeeze(-1)]                  # 选中组件均值
        selected_component_std = component_std_devs[batch_indices, selected_component_idx.squeeze(-1)]                # 选中组件标准差
        # 重参数化采样：a = μ + σ * ε
        random_noise = torch.randn((batch_size, self.action_dimension), device=obs.device)
        sampled_action = selected_component_mean + selected_component_std * random_noise
        # 计算动作概率
        component_log_probs = create_log_gaussian(component_means, constrained_log_std, sampled_action.unsqueeze(1))  # 各组件概率
        # log_p_x_t = torch.logsumexp(component_log_probs + log_mixture_weights, dim=1)                               # 边际概率
        # log_p_x_t -= torch.logsumexp(log_mixture_weights, dim=1)                                                    # 归一化
        log_prob_numerator = torch.logsumexp(component_log_probs + log_mixture_weights, dim=1)                        # 分子
        log_prob_denominator = torch.logsumexp(log_mixture_weights, dim=1)                                            # 分母
        log_probability = log_prob_numerator - log_prob_denominator                                                   # 非原地操作
        return selected_component_mean, sampled_action, log_probability


class MLPActorCritic(nn.Module):
    """
    MLPActorCritic is an actor-critic neural network module for reinforcement learning, combining both policy (actor) and value (critic) networks. 

        state_dim (int): Dimension of the input state space.
        action_dim (int): Dimension of the action space.
        hidden_sizes (tuple, optional): Sizes of hidden layers for MLPs. Default is (128, 128, 128, 128).
        activation (callable, optional): Activation function for hidden layers. Default is F.leaky_relu.
        output_activation (callable, optional): Activation function for output layers. Default is None.

    Attributes:
        state_dimension (int): Dimension of the input state space.
        action_dimension (int): Dimension of the action space.
        hidden_activation (callable): Activation function for hidden layers.
        policy_network (MLPGaussianPolicy): Actor network that outputs a Gaussian policy.
        shared_cnn_dense (CNNDense): Shared CNN-based feature extractor for critic networks.
        q_network_1 (MLP): First Q-value critic network.
        q_network_2 (MLP): Second Q-value critic network.

    Methods:
        forward(state_input, action_input=None):
    """
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_sizes=(128, 128, 128, 128),
            activation=F.leaky_relu,
            output_activation=None
            ):
        super(MLPActorCritic, self).__init__()

        self.state_dimension = state_dim
        self.action_dimension = action_dim
        self.hidden_activation = activation

        # ============= Actor网络(策略网络) =============
        self.policy_network = MLPGaussianPolicy(
            state_dim,
            action_dim,
            list(hidden_sizes),
            activation,
            output_activation
            )

        # ============= Critic网络(价值网络) =============
        # 共享CNN特征提取器
        self.shared_cnn_dense = CNNDense(activation, None)

        # Q网络：输入特征+动作，输出Q值
        q_network_input_dim = 128 + 10 + action_dim  # CNN特征 138维 + 动作 2维 = 140维
        self.q_network_1 = MLP(q_network_input_dim, list(hidden_sizes) + [1], activation, None)
        self.q_network_2 = MLP(q_network_input_dim, list(hidden_sizes) + [1], activation, None)

    def forward(self, state_input, action_input=None):
        """
        Performs a forward pass through the actor-critic network.
        Args:
            state_input (torch.Tensor): Input state tensor.
            action_input (torch.Tensor, optional): Action tensor. If provided, computes Q-values for the given actions.
        Returns:
            Tuple:
                If `action_input` is provided:
                    squashed_mean (torch.Tensor): Mean of the action distribution after squashing.
                    squashed_action (torch.Tensor): Sampled action after squashing.
                    adjusted_log_prob (torch.Tensor): Log-probability of the sampled action.
                    q1_value (torch.Tensor): Q-value from the first critic for (state, action).
                    q2_value (torch.Tensor): Q-value from the second critic for (state, action).
                    q1_policy_value (torch.Tensor): Q-value from the first critic for (state, policy action).
                    q2_policy_value (torch.Tensor): Q-value from the second critic for (state, policy action).
                If `action_input` is not provided:
                    squashed_mean (torch.Tensor): Mean of the action distribution after squashing.
                    squashed_action (torch.Tensor): Sampled action after squashing.
                    adjusted_log_prob (torch.Tensor): Log-probability of the sampled action.
                    q1_policy_value (torch.Tensor): Q-value from the first critic for (state, policy action).
                    q2_policy_value (torch.Tensor): Q-value from the second critic for (state, policy action).
        Notes:
            - The actor network outputs are squashed using a tanh function to ensure actions are within bounds.
            - Critic networks estimate Q-values for both the policy action and optionally provided actions.
        """

        # ============= Actor网络(策略网络) =============
        action_mean, sampled_action, log_action_prob = self.policy_network(state_input)
        # 应用tanh压缩到有界动作空间
        squashed_mean, squashed_action, adjusted_log_prob = apply_squashing_func(action_mean, sampled_action, log_action_prob)
        # ============= Critic网络(价值网络) =============
        # 共享CNN特征提取
        extracted_features = self.shared_cnn_dense(state_input)  # 提取136维特征
        # 计算当前策略下的Q值
        q1_policy_value = self.q_network_1(torch.cat([extracted_features, squashed_action], dim=-1)).squeeze(-1)    # Q1(s,π(s))
        q2_policy_value = self.q_network_2(torch.cat([extracted_features, squashed_action], dim=-1)).squeeze(-1)    # Q2(s,π(s))
        if action_input is not None:
            # 如果提供了动作，计算Q(s,a)
            q1_value = self.q_network_1(torch.cat([extracted_features, action_input], dim=-1)).squeeze(-1)        # Q1(s,a)
            q2_value = self.q_network_2(torch.cat([extracted_features, action_input], dim=-1)).squeeze(-1)        # Q2(s,a)
            return squashed_mean, squashed_action, adjusted_log_prob, q1_value, q2_value, q1_policy_value, q2_policy_value
        else:
            return squashed_mean, squashed_action, adjusted_log_prob, q1_policy_value, q2_policy_value