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
        super().__init__()
        self.action_dimension = action_dim
        self.num_mixture_components = 4
        self.hidden_size = [128, 128, 128, 128]
        self.alpha_activation_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.net = MLP(
            input_dim=138, # the flattened_features has 128 elements, and the rest of the observation has 10 elements (280 - 270)
            hidden_sizes=self.hidden_size,
            activation=F.leaky_relu,
            output_activation=F.leaky_relu,
        )
        self.gmm_output_layer = nn.Linear(self.hidden_size[-1], (action_dim * 2 + 1) * self.num_mixture_components)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [n_env * batch_size, n_obs] = [1024 * 32, 280] in each row the first 270 elements are LiDAR data
        batch_size = obs.shape[0]
        num_lidar_points = 90  # 270/3
        lidar_raw = obs[:, :270].reshape(batch_size, num_lidar_points, 3)
        # 处理距离信息(第3个特征) - 使用简单的激活函数替代
        lidar_raw[:, :, 2] = torch.clamp(1.0 / (lidar_raw[:, :, 2] + self.alpha_activation_param + 1e-8), 
                                        min=1e-6, max=1e6)

        cnn = getattr(self, 'cnn', None)
        if cnn is None:
            self.cnn = CNNNet().to(obs.device)
            cnn = self.cnn
        flattened_features = cnn(lidar_raw)
        obs_rest = obs[:, 270:]  # [batch_size, n_obs-270]
        obs_new = torch.cat([flattened_features, obs_rest], dim=1)
        
        # 直接处理观测，不需要动作输入
        x = self.net(obs_new)
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
        
        # 选择组件
        selected_component_idx = torch.multinomial(torch.softmax(log_mixture_weights, dim=-1), num_samples=1)
        
        # 获取选中组件的参数
        batch_indices = torch.arange(batch_size, device=obs.device)
        selected_component_mean = component_means[batch_indices, selected_component_idx.squeeze(-1)]  # 选中组件均值
        selected_component_std = component_std_devs[batch_indices, selected_component_idx.squeeze(-1)]  # 选中组件标准差
        
        # 重参数化采样：a = μ + σ * ε
        random_noise = torch.randn((batch_size, self.action_dimension), device=obs.device)
        sampled_action = selected_component_mean + selected_component_std * random_noise
        
        # 计算动作概率
        component_log_probs = create_log_gaussian(component_means, constrained_log_std, sampled_action.unsqueeze(1))  # 各组件概率
        log_prob_numerator = torch.logsumexp(component_log_probs + log_mixture_weights, dim=1)  # 分子
        log_prob_denominator = torch.logsumexp(log_mixture_weights, dim=1)  # 分母
        log_probability = log_prob_numerator - log_prob_denominator  # 非原地操作
        
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
        # 使用自定义的CNN特征提取器(匹配270维LiDAR输入)
        self.shared_cnn_dense = self._create_cnn_feature_extractor(activation)

        # Q网络：输入特征+动作，输出Q值
        q_network_input_dim = 128 + 10 + action_dim  # CNN特征 138维 + 动作 2维 = 140维
        self.q_network_1 = MLP(q_network_input_dim, list(hidden_sizes) + [1], activation, None)
        self.q_network_2 = MLP(q_network_input_dim, list(hidden_sizes) + [1], activation, None)

    def _create_cnn_feature_extractor(self, activation):
        """创建与270维LiDAR输入兼容的CNN特征提取器"""
        class CNN270FeatureExtractor(nn.Module):
            def __init__(self, activation):
                super().__init__()
                self.alpha_activation_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
                # 对于270维输入(90个点，每个点3个特征)
                self.conv_layer_1 = nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0)
                self.conv_layer_2 = nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0)
                self.conv_layer_3 = nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0)
                self.max_pool = nn.MaxPool1d(kernel_size=90, stride=1, padding=0)
                self.activation_func = activation
                
            def forward(self, input_data):
                # 提取LiDAR数据（前270维）并重塑
                lidar_data = input_data[:, 0:270]  # [batch, 270]
                reshaped_lidar = lidar_data.view(-1, 90, 3)  # [batch, 90方向, 3特征]
                
                # 处理距离信息(第3个特征) - 使用简单的激活函数替代
                processed_distance = torch.clamp(1.0 / (reshaped_lidar[:, :, 2] + self.alpha_activation_param + 1e-8), 
                                                min=1e-6, max=1e6)
                
                # 重新组合特征
                combined_lidar_features = torch.cat([
                    reshaped_lidar[:, :, 0:2],  # 方向信息
                    processed_distance.unsqueeze(-1),  # 处理后的距离
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
                
                # 拼接CNN特征和其他状态信息（后10维）
                return torch.cat([flattened_cnn_features, input_data[:, 270:]], dim=-1)
                
        return CNN270FeatureExtractor(activation)

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
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, hidden_sizes=(128, 128, 128, 128), device='cuda'):
        """
        Initialize DCLP algorithm
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            lr: Learning rate
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
        
        # Initialize networks
        self.actor_critic = MLPActorCritic(state_dim, action_dim, hidden_sizes).to(device)
        self.target_actor_critic = MLPActorCritic(state_dim, action_dim, hidden_sizes).to(device)
        
        # Copy parameters to target network
        self.update_target_network(tau=1.0)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.policy_network.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.actor_critic.shared_cnn_dense.parameters()) + 
            list(self.actor_critic.q_network_1.parameters()) + 
            list(self.actor_critic.q_network_2.parameters()), lr=lr
        )
        
    def get_action(self, state, deterministic=False):
        """Get action from current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                # Use mean action for deterministic policy
                action_mean, _, _ = self.actor_critic.policy_network(state_tensor)
                return action_mean.cpu().numpy()[0]
            else:
                # Sample action from policy
                _, sampled_action, _ = self.actor_critic.policy_network(state_tensor)
                return sampled_action.cpu().numpy()[0]
                
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
            _, next_actions, next_log_probs = self.target_actor_critic.policy_network(next_states)
            
            # Apply squashing and get target Q-values
            _, next_actions_squashed, next_log_probs_adjusted = apply_squashing_func(
                next_actions, next_actions, next_log_probs
            )
            
            # Get target Q-values
            target_features = self.target_actor_critic.shared_cnn_dense(next_states)
            target_q1 = self.target_actor_critic.q_network_1(
                torch.cat([target_features, next_actions_squashed], dim=-1)
            ).squeeze(-1)
            target_q2 = self.target_actor_critic.q_network_2(
                torch.cat([target_features, next_actions_squashed], dim=-1)
            ).squeeze(-1)
            
            # Take minimum and subtract entropy term
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs_adjusted
            target_q = rewards + (~dones) * self.gamma * target_q
            
        # Current Q-values
        current_features = self.actor_critic.shared_cnn_dense(states)
        current_q1 = self.actor_critic.q_network_1(
            torch.cat([current_features, actions], dim=-1)
        ).squeeze(-1)
        current_q2 = self.actor_critic.q_network_2(
            torch.cat([current_features, actions], dim=-1)
        ).squeeze(-1)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ============= Actor Update =============
        # Get current policy actions and Q-values
        _, policy_actions, log_probs = self.actor_critic.policy_network(states)
        _, policy_actions_squashed, log_probs_adjusted = apply_squashing_func(
            policy_actions, policy_actions, log_probs
        )
        
        # Q-values for policy actions
        policy_q1 = self.actor_critic.q_network_1(
            torch.cat([current_features.detach(), policy_actions_squashed], dim=-1)
        ).squeeze(-1)
        policy_q2 = self.actor_critic.q_network_2(
            torch.cat([current_features.detach(), policy_actions_squashed], dim=-1)
        ).squeeze(-1)
        
        policy_q = torch.min(policy_q1, policy_q2)
        
        # Actor loss (negative because we want to maximize Q)
        actor_loss = (self.alpha * log_probs_adjusted - policy_q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target network
        self.update_target_network()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'target_q_mean': target_q.mean().item()
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