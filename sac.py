import torch
import torch.nn as nn
from torch.nn import Linear, Module, Sequential, SiLU
from torch.distributions import Normal
from torch.nn import functional as F

class EnvModelWithSAC(Module):
    def __init__(
        self, action_space, observation_space, hidden_size=64, hidden_layers=4
    ):
        super().__init__()

        # Shared input and hidden layers (original EnvModel structure)
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        input_len = action_dim + obs_dim
        
        self._input = Linear(input_len, hidden_size)
        
        hidden_layers_list = [
            [Linear(hidden_size, hidden_size), SiLU()]
            for _ in range(hidden_layers)
        ]
        self._hidden = Sequential(
            *[layer for tier in hidden_layers_list for layer in tier]
        )
        
        # Original env model head (dynamics prediction)
        self._output = Linear(hidden_size, obs_dim)
        
        # Actor head (policy network)
        # Takes only observation, outputs action distribution parameters
        self.actor_input = Linear(obs_dim, hidden_size)
        self.actor_mean = Linear(hidden_size, action_dim)
        self.actor_log_std = Linear(hidden_size, action_dim)
        
        # Critic heads (two Q-networks)
        # Takes observation + action, outputs Q-value
        self.critic1_output = Linear(hidden_size, 1)
        self.critic2_output = Linear(hidden_size, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, *args):
        """Original forward pass for env model (dynamics prediction)"""
        if len(args) == 1:
            return self._batch_forward(*args)
        elif len(args) == 2:
            return self._one_forward(*args)

    def _batch_forward(self, X):
        """Original batch forward for dynamics model"""
        out = self._input(X.float())
        out = self._hidden(out)
        out = self._output(out)
        return out

    def _one_forward(self, observation, action):
        """Original one forward for dynamics model"""
        inp = torch.cat([action._tensor, observation._tensor], dim=-1)
        return self._batch_forward(inp)
    
    def _get_shared_features(self, obs, action):
        """
        Get shared features from the hidden layers.
        This is where the magic happens - reusing _hidden for actor/critic.
        """
        inp = torch.cat([action, obs], dim=-1)
        features = self._input(inp.float())
        features = self._hidden(features)
        return features
    
    def forward_actor(self, obs):
        """
        Forward pass through actor network.
        Actor only needs observation, not action.
        """
        # Actor has its own input layer since it only takes observation
        features = self.actor_input(obs.float())
        features = self._hidden(features)  # Reuse shared hidden layers
        
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def forward_critic(self, obs, action):
        """
        Forward pass through both critic networks.
        Critics take observation + action and use shared features.
        """
        features = self._get_shared_features(obs, action)
        q1 = self.critic1_output(features)
        q2 = self.critic2_output(features)
        return q1, q2
    
    def sample_action(self, obs, deterministic=False):
        """
        Sample an action from the policy.
        Uses reparameterization trick for backpropagation.
        """
        mean, log_std = self.forward_actor(obs)
        
        if deterministic:
            return torch.tanh(mean), None
        
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def predict_next_state(self, obs, action):
        """Predict state delta using the dynamics model"""
        return self._one_forward(obs, action)


class SACAgent:
    def __init__(self, observation_space, action_space, hidden_size=256, hidden_layers=3):
        self.network = EnvModelWithSAC(
            action_space, observation_space, hidden_size, hidden_layers
        )
        
        # Separate optimizers for different components
        # Env model optimizer (for dynamics learning)
        self.env_model_params = (
            list(self.network._input.parameters()) +
            list(self.network._hidden.parameters()) +
            list(self.network._output.parameters())
        )
        self.env_model_optimizer = torch.optim.Adam(self.env_model_params, lr=1e-3)
        
        # Actor optimizer
        self.actor_params = (
            list(self.network.actor_input.parameters()) +
            list(self.network.actor_mean.parameters()) +
            list(self.network.actor_log_std.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=3e-4)
        
        # Critic optimizer (includes shared hidden layers)
        self.critic_params = (
            list(self.network.critic1_output.parameters()) +
            list(self.network.critic2_output.parameters())
        )
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=3e-4)
        
        # Target network
        self.target_network = EnvModelWithSAC(
            action_space, observation_space, hidden_size, hidden_layers
        )
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
    
    def update_env_model(self, states, actions, state_deltas):
        """Update the dynamics model"""
        predicted_deltas = self.network(states, actions)
        loss = F.mse_loss(predicted_deltas, state_deltas)
        
        self.env_model_optimizer.zero_grad()
        loss.backward()
        self.env_model_optimizer.step()
        
        return loss.item()
    
    def update_critics(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """Update critic networks"""
        with torch.no_grad():
            next_actions, next_log_probs = self.network.sample_action(next_states)
            target_q1, target_q2 = self.target_network.forward_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            alpha = self.log_alpha.exp()
            target_q = target_q - alpha * next_log_probs
            target = rewards + (1 - dones) * gamma * target_q
        
        q1, q2 = self.network.forward_critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, states, target_entropy=-2):
        """Update actor network"""
        actions, log_probs = self.network.sample_action(states)
        q1, q2 = self.network.forward_critic(states, actions)
        q_value = torch.min(q1, q2)
        
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_probs - q_value).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return actor_loss.item(), alpha_loss.item()
    
    def soft_update_target(self, tau=0.005):
        """Soft update of target network"""
        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
