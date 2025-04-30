import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any
from gym import spaces


class FeatureExtractor(nn.Module):
    """Feature extractor for agent observations"""
    def __init__(self, fov_size: int):
        super(FeatureExtractor, self).__init__()
        self.fov_size = fov_size
        
        # Local view and agent view feature extraction
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate output size after convolutions and pooling
        conv_output_size = ((fov_size // 2) ** 2) * 32
        
        # FC layer for CNN output
        self.fc1 = nn.Linear(conv_output_size, 128)
        
        # FC layer for vector features (goal_vec, wp_vec, distances)
        self.fc2 = nn.Linear(7, 32)  # 2 (goal_vec) + 2 (wp_vec) + 3 (distances) = 7
        
        # Combined feature representation
        self.fc_combined = nn.Linear(128 + 32, 128)
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract and process grid-based features
        local_view = obs['local_view'].unsqueeze(1)  # Add channel dimension
        other_agents = obs['other_agents'].unsqueeze(1)  # Add channel dimension
        
        # Combine grid inputs
        grid_input = torch.cat([local_view, other_agents], dim=1)
        
        # CNN feature extraction
        x1 = F.relu(self.conv1(grid_input))
        x1 = self.pool(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.pool(x1)
        x1 = x1.view(x1.size(0), -1)  # Flatten
        x1 = F.relu(self.fc1(x1))
        
        # Process vector features
        goal_vec = obs['goal_vec']
        wp_vec = obs['wp_vec']
        distances = obs['distances']
        
        # Combine vector inputs
        vector_input = torch.cat([goal_vec, wp_vec, distances], dim=1)
        x2 = F.relu(self.fc2(vector_input))
        
        # Combine all features
        combined = torch.cat([x1, x2], dim=1)
        features = F.relu(self.fc_combined(combined))
        
        return features


class MAPFAgent(nn.Module):
    """
    Agent for Multi-Agent Path Finding using PPO
    """
    def __init__(self, obs_space: spaces.Dict, num_actions: int):
        super(MAPFAgent, self).__init__()
        
        # Get observation space details
        self.fov_size = obs_space['local_view'].shape[0]
        self.num_actions = num_actions
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(self.fov_size)
        
        # Policy head
        self.policy = nn.Linear(128, num_actions)
        
        # Value head
        self.value = nn.Linear(128, 1)
        
        # Initialize optimizer (Adam with standard PPO hyperparameters)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, eps=1e-5)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def preprocess_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy observations to PyTorch tensors"""
        processed_obs = {}
        for key, value in obs.items():
            processed_obs[key] = torch.FloatTensor(value)
            # Add batch dimension if missing
            if len(processed_obs[key].shape) == 1:
                processed_obs[key] = processed_obs[key].unsqueeze(0)
        return processed_obs
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        features = self.feature_extractor(obs)
        action_logits = self.policy(features)
        value = self.value(features)
        return action_logits, value
    
    def select_action(self, obs: Dict[str, np.ndarray]) -> Tuple[int, float, float]:
        """
        Select an action using the current policy
        Returns: action, log_prob, value
        """
        # Convert observations to tensors
        with torch.no_grad():
            processed_obs = self.preprocess_obs(obs)
            logits, value = self(processed_obs)
            
            # Sample action from categorical distribution
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update
        Returns: log_probs, values, entropy
        """
        logits, values = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        return log_probs, values.squeeze(), entropy
    
    def train_on_batch(self, batch, weights=None):
        """
        Train on a batch of experiences using PPO
        """
        # Unpack batch
        states, actions, old_log_probs, returns, advantages = batch
        
        # Process states
        processed_states = {}
        for key in states[0].keys():
            processed_states[key] = torch.stack([torch.FloatTensor(s[key]) for s in states])
        
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(old_log_probs)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        
        if weights is not None:
            weights = torch.FloatTensor(weights)
        
        # PPO clip parameter
        clip_range = 0.2
        
        # Get current log probs and values
        log_probs, values, entropy = self.evaluate_actions(processed_states, actions)
        
        # Calculate ratio (π_θ / π_θold)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO losses
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # Value function loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_coef = 0.01
        entropy_loss = -entropy_coef * entropy
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # Apply importance sampling weights if provided
        if weights is not None:
            loss = loss * weights.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (commonly used in PPO)
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item()