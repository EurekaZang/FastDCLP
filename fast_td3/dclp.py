import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dclp_utils import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'DCLP'))
from dclp_utils import CNNNet, CNNDense, MLP

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
        super(MLPGaussianPolicy, self).__init__()
        self.num_mixture_components = 4  # 高斯混合组件数量
        self.action_dimension = action_dim
        self.hidden_activation = activation
        self.output_activation = output_activation
        # CNN自适应激活参数
        self.alpha_activation_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # CNN网络
        self.cnn_feature_extractor = CNNNet()
        # MLP网络
        # CNN输出128维 + 其他状态信息10维 = 138维
        self.mlp_network = MLP(138, hidden_sizes, activation, activation)
        # GMM参数输出层
        self.gmm_output_layer = nn.Linear(hidden_sizes[-1], (self.action_dimension*2+1)*self.num_mixture_components)

    def forward(self, state_input):
        """
        Forward pass of the model.

        Args:
            state_input (torch.Tensor): Input tensor of shape [batch_size, input_dim]. The input contains concatenated lidar features,
                additional state information, and possibly other features.

        Returns:
            tuple:
                - selected_component_mean (torch.Tensor): The mean of the selected Gaussian component for each sample, shape [batch_size, action_dimension].
                - sampled_action (torch.Tensor): The sampled action from the selected Gaussian component, shape [batch_size, action_dimension].
                - log_probability (torch.Tensor): The log-probability of the sampled action under the Gaussian Mixture Model, shape [batch_size].
        
        Workflow:
            1. Processes lidar data and applies a custom activation to the distance feature.
            2. Extracts additional state information.
            3. Extracts features using a CNN and concatenates with other features.
            4. Processes concatenated features with an MLP.
            5. Generates GMM parameters (weights, means, log-stds) for each mixture component.
            6. Applies constraints to log-stds for numerical stability.
            7. Samples a Gaussian component and draws an action using the reparameterization trick.
            8. Computes the log-probability of the sampled action under the GMM.
        """

        batch_size = state_input.shape[0]
        # 处理激光雷达数据
        lidar_data = state_input[:, 0:6*90]
        reshaped_lidar = lidar_data.view(-1, 90, 6)
        # 对距离特征应用自定义激活
        processed_distance = reciprocal_relu(reshaped_lidar[:, :, 2], self.alpha_activation_param)
        combined_lidar_features = torch.cat([
            reshaped_lidar[:, :, 0:2],
            processed_distance.unsqueeze(-1),
            reshaped_lidar[:, :, 3:6]
        ], dim=-1)
        # 提取其他状态信息, not used in this context
        additional_state_info = state_input[:, 6*90:6*90+8]
        additional_state_info = additional_state_info.view(-1, 8)
        # CNN特征提取
        cnn_features = self.cnn_feature_extractor(combined_lidar_features, additional_state_info)
        # 特征拼接
        concatenated_features = torch.cat([cnn_features, state_input[:, 6*90:]], dim=-1)
        # MLP处理
        mlp_output = self.mlp_network(concatenated_features)
        # 生成GMM参数：(权重 + 均值 + 对数标准差) × k个组件
        gmm_parameters = self.gmm_output_layer(mlp_output)
        reshaped_gmm_params = gmm_parameters.view(-1, self.num_mixture_components, 2*self.action_dimension+1)
        # 分离参数
        log_mixture_weights = reshaped_gmm_params[..., 0]              # 混合权重 [N, K]
        component_means = reshaped_gmm_params[..., 1:1+self.action_dimension]       # 均值 [N, K, action_dimension]
        log_component_std = reshaped_gmm_params[..., 1+self.action_dimension:]   # 对数标准差 [N, K, action_dimension]
        # 约束对数标准差到安全范围
        constrained_log_std = torch.tanh(log_component_std)
        constrained_log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (constrained_log_std + 1)
        component_std_devs = torch.exp(constrained_log_std)
        # 采样高斯组件
        selected_component_idx = torch.multinomial(torch.softmax(log_mixture_weights, dim=-1), num_samples=1)  # 选择组件
        # 获取选中组件的参数
        batch_indices = torch.arange(batch_size, device=state_input.device)
        selected_component_mean = component_means[batch_indices, selected_component_idx.squeeze(-1)]      # 选中组件均值
        selected_component_std = component_std_devs[batch_indices, selected_component_idx.squeeze(-1)] # 选中组件标准差
        # 重参数化采样：a = μ + σ * ε
        random_noise = torch.randn((batch_size, self.action_dimension), device=state_input.device)
        sampled_action = selected_component_mean + selected_component_std * random_noise
        # 计算动作概率
        component_log_probs = create_log_gaussian(component_means, constrained_log_std, sampled_action.unsqueeze(1))  # 各组件概率
        # log_p_x_t = torch.logsumexp(component_log_probs + log_mixture_weights, dim=1)      # 边际概率
        # log_p_x_t -= torch.logsumexp(log_mixture_weights, dim=1)                   # 归一化
        log_prob_numerator = torch.logsumexp(component_log_probs + log_mixture_weights, dim=1)    # 分子
        log_prob_denominator = torch.logsumexp(log_mixture_weights, dim=1)               # 分母  
        log_probability = log_prob_numerator - log_prob_denominator              # 非原地操作
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
            output_activation=F.leaky_relu
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
        # 使用自定义的CNN特征提取器(匹配540维LiDAR输入，90个点*6个特征)
        self.shared_cnn_dense = CNNDense(activation, None)

        # Q网络：输入特征+动作，输出Q值
        # CNN特征 128维 + 其他观测特征 (总观测维度 - 540) + 动作维度
        other_obs_dim = state_dim - 540  # 其他观测特征的维度
        q_network_input_dim = 128 + other_obs_dim + action_dim
        self.q_network_1 = MLP(q_network_input_dim, list(hidden_sizes) + [1], activation, None)
        self.q_network_2 = MLP(q_network_input_dim, list(hidden_sizes) + [1], activation, None)
        
        # 确保两个Q网络有不同的初始化
        self._initialize_networks()

    def _initialize_networks(self):
        for layer in self.q_network_1.layer_modules:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        for layer in self.q_network_2.layer_modules:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

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


class DCLP:
    """
    Distributional Categorical Learning Policy (DCLP) Algorithm
    
    This class implements the DCLP training algorithm for reinforcement learning
    with distributional Q-learning and categorical policy networks.
    """

    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, hidden_sizes=(128, 128, 128, 128), device='cuda'):
        """
        Initialize DCLP algorithm
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Temperature parameter for entropy regularization
            hidden_sizes: Hidden layer sizes for networks
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.actor_critic = MLPActorCritic(state_dim, action_dim, hidden_sizes).to(device)
        self.target_actor_critic = MLPActorCritic(state_dim, action_dim, hidden_sizes).to(device)
        self.update_target_network(tau=self.tau)
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.policy_network.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.actor_critic.shared_cnn_dense.parameters()) +
            list(self.actor_critic.q_network_1.parameters()) +
            list(self.actor_critic.q_network_2.parameters()), lr=critic_lr
        )

    def get_action(self, state, deterministic=True):
        """Get action from current policy"""
        with torch.no_grad():
            if deterministic:
                # Use mean action for deterministic policy
                action_mean, _, _, _, _ = self.actor_critic(state)
                # action = torch.tanh(action_mean)
                print(f"Deterministic action mean: {action_mean}")
                return action_mean
            else:
                # Sample action from policy
                _, sampled_action, _, _, _ = self.actor_critic(state)
                print(f"Stochastic sampled action: {sampled_action}")
                return sampled_action

    def update_target_network(self, tau=None):
        """Soft update of target network parameters"""
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_actor_critic.parameters(),
                                     self.actor_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_step(self, batch):
        """
        Single training step
        Args:
            batch: Dictionary containing 'state', 'action', 'reward', 'next_state', 'done'
        """
        states = torch.FloatTensor(batch['state']).to(self.device)
        actions = torch.FloatTensor(batch['action']).to(self.device)
        rewards = torch.FloatTensor(batch['reward']).to(self.device)
        next_states = torch.FloatTensor(batch['next_state']).to(self.device)
        dones = torch.BoolTensor(batch['done']).to(self.device)

        # ============= Critic Update =============
        with torch.no_grad():
            # Target actions from target policy
            _, _, next_log_probs, target_q1, target_q2 = self.target_actor_critic(next_states)

            # Take minimum and subtract entropy term
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            # target_q = rewards + (~dones) * self.gamma * target_q
            target_q = rewards + self.gamma * target_q

        # Current Q-values
        _, _, _, current_q1, current_q2, _, _ = self.actor_critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 冻结Q网络参数，只更新策略网络
        for param in self.actor_critic.q_network_1.parameters():
            param.requires_grad = False
        for param in self.actor_critic.q_network_2.parameters():
            param.requires_grad = False
        for param in self.actor_critic.shared_cnn_dense.parameters():
            param.requires_grad = False

        # ============= Actor Update =============
        # Get current policy actions and Q-values
        _, _, log_probs, _, _ = self.actor_critic(states)
        with torch.no_grad():
            _, _, _, policy_q1, policy_q2 = self.actor_critic(states)

        policy_q = torch.min(policy_q1, policy_q2)

        # Actor loss (negative because we want to maximize Q)
        actor_loss = (self.alpha * log_probs - policy_q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.actor_critic.q_network_1.parameters():
            param.requires_grad = True
        for param in self.actor_critic.q_network_2.parameters():
            param.requires_grad = True
        for param in self.actor_critic.shared_cnn_dense.parameters():
            param.requires_grad = True

        # Update target network
        self.update_target_network(tau=self.tau)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'target_q_mean': target_q.mean().item(),
            'policy_q1_mean': policy_q1.mean().item(),
            'policy_q2_mean': policy_q2.mean().item(),
            'policy_q_mean': policy_q.mean().item(),
            'log_probs_mean': log_probs.mean().item(),
            'entropy_term': (self.alpha * log_probs).mean().item(),
            'alpha': self.alpha
        }

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'target_actor_critic': self.target_actor_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.target_actor_critic.load_state_dict(checkpoint['target_actor_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])