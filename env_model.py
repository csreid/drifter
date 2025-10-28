import torch
from tqdm import tqdm
from collections import deque
from torch.nn import (
	Linear,
	Module,
	Sequential,
	LeakyReLU,
	MSELoss,
	Sigmoid,
	SiLU,
)
from torch.optim import Adam, SGD
from torch.nn import functional as F
import numpy as np
from exploration_policy import ExplorationPolicy
from tabulate import tabulate
from IPython import embed
from drifter_env import observation_space, action_space
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from memory import Transition, State, Action
from batched_memory import (
	BatchedState,
	BatchedTransition,
	BatchedAction,
	BatchedStateDelta,
)
from torch.utils.tensorboard import SummaryWriter
from collections import deque

def collate_fn(batch):
	actions = [b[0] for b in batch]
	obs = [b[1] for b in batch]
	deltas = [b[2] for b in batch]

	return (
		BatchedAction.from_list(actions),
		BatchedState.from_list(obs),
		BatchedStateDelta.from_list(deltas),
	)


class EnvModel(Module):
	def __init__(
		self, action_space, observation_space, hidden_size=64, hidden_layers=4
	):
		super().__init__()

		# Assume the input spaces are 1-d, this will break otherwise
		print(f"Action space shape: {action_space.shape}")
		print(f"Observation space shape: {observation_space.shape}")
		input_len = action_space.shape[0] + observation_space.shape[0]
		print(f"Total size: {input_len}")
		self._input = Linear(input_len, hidden_size)

		hidden_layers = [
			[Linear(hidden_size, hidden_size), SiLU()]
			for _ in range(hidden_layers)
		]
		self._hidden = Sequential(
			*[layer for tier in hidden_layers for layer in tier]
		)
		self._output = Linear(hidden_size, observation_space.shape[0])

		EnvModel._initialize(self)

	@staticmethod
	def _initialize(m):
		if isinstance(m, (Linear)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, *args):
		"""If called with one argument, that argument should
		be an appropriately-shaped batch of action/observation
		tensors

		If called with two arguments, they should be a single action and state pair
		"""
		if len(args) == 1:
			return self._batch_forward(*args)
		elif len(args) == 2:
			return self._one_forward(*args)

	def _batch_forward(self, X):
		out = self._input(X.float())
		out = self._hidden(out)
		# out = F.tanh(out)
		out = self._output(out)

		return out

	def _one_forward(self, observation, action):
		inp = torch.cat([action._tensor, observation._tensor], dim=-1)

		return self._batch_forward(inp)


def loss_fn(inp: BatchedStateDelta, targets: BatchedStateDelta):
	loss_components = {
		"position": F.mse_loss(inp.position._tensor, targets.position._tensor),
#		"goal_position": F.mse_loss(
#			inp.local_goal_position._tensor, targets.local_goal_position._tensor
#		),
		"absolute_goal_position": F.mse_loss(
			inp.absolute_goal_position._tensor,
			targets.absolute_goal_position._tensor,
		),
		"velocity": F.mse_loss(inp.velocity._tensor, targets.velocity._tensor),
		# "is_flipped": F.binary_cross_entropy_with_logits(
		# inp.is_flipped, targets.is_flipped
		# ),
		# "omega": F.mse_loss(omega_inp, omega_target),
		"orientation": F.mse_loss(
			inp.orientation._tensor, targets.orientation._tensor
		),
	}

	loss_weights = {
		"position": 1.0,
		#"goal_position": 1.0,
		"absolute_goal_position": 1.0,
		"velocity": 1.0,
		# "is_flipped": 100.0,
		# "omega": 1.0,
		"orientation": 1.0,
	}

	loss_val = torch.sum(
		torch.stack(
			[
				val * weight
				for [(_, val), (_, weight)] in zip(
					loss_components.items(), loss_weights.items()
				)
			]
		)
	)

	return loss_val, loss_components


class TransitionDataset(Dataset):
	def __init__(self, df):
		self._df = df

	def __len__(self):
		return len(self._df)

	def __getitem__(self, idx):
		item = self._df.iloc[idx]

		obs = State.from_tensor(
			torch.tensor(
				[
					item.pos_x,
					item.pos_y,
					item.pos_z,
					item.goal_pos_x,
					item.goal_pos_y,
					item.goal_pos_z,
					item.vel_x,
					item.vel_y,
					item.vel_z,
					item.is_flipped,
					item.absolute_goal_pos_x,
					item.absolute_goal_pos_y,
					item.absolute_goal_pos_z,
					item.orn_x,
					item.orn_y,
					item.orn_z,
					item.orn_w,
				]
			).float()
		)

		next_obs = State.from_tensor(
			torch.tensor(
				[
					item.next_pos_x,
					item.next_pos_y,
					item.next_pos_z,
					item.next_goal_pos_x,
					item.next_goal_pos_y,
					item.next_goal_pos_z,
					item.next_vel_x,
					item.next_vel_y,
					item.next_vel_z,
					item.next_is_flipped,
					item.next_absolute_goal_pos_x,
					item.next_absolute_goal_pos_y,
					item.next_absolute_goal_pos_z,
					item.next_orn_x,
					item.next_orn_y,
					item.next_orn_z,
					item.next_orn_w,
				]
			).float()
		)

		action = Action.from_tensor(
			torch.tensor([item.action_0, item.action_1])
		)

		delta = next_obs - obs

		return action, obs, delta


def main():
	use_gpu = torch.cuda.is_available()
	dev = "cuda:0" if use_gpu else "cpu"

	writer = SummaryWriter()
	env_model = EnvModel(
		action_space, observation_space, hidden_size=512, hidden_layers=2
	).to(dev)
	env_model.load_state_dict(torch.load("model.pt", weights_only=True))
	opt = Adam(env_model.parameters())
	df = pd.read_csv("transitions.csv")
	train_df = df.sample(frac=0.8)
	test_df = df.drop(train_df.index)

	train_ds = TransitionDataset(train_df)
	test_ds = TransitionDataset(test_df)

	train_dataloader = DataLoader(
		train_ds, batch_size=1024, shuffle=True, collate_fn=collate_fn
	)
	test_dataloader = DataLoader(
		test_ds, batch_size=1024, shuffle=True, collate_fn=collate_fn
	)

	for epoch in tqdm(range(50)):
		for i, (action_X, state_X, batch_Y) in tqdm(
			enumerate(train_dataloader),
			leave=False,
			total=len(train_dataloader),
		):

			action_X = action_X.to(dev)
			state_X = state_X.to(dev)
			batch_Y = batch_Y.to(dev)

			pred_X = BatchedStateDelta.from_tensor(
				env_model(state_X, action_X)
			)
			loss, per_output_loss = loss_fn(pred_X, batch_Y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			writer.add_scalars(
				"loss_components",
				per_output_loss,
				epoch * len(train_dataloader) + i,
			)
			writer.add_scalar(f"Loss", loss, epoch * len(train_dataloader) + i)

			with torch.no_grad():
				action_test_X, obs_test_X, test_Y = next(iter(test_dataloader))

				test_pred_X = BatchedStateDelta.from_tensor(
					env_model(obs_test_X, action_test_X)
				)
				test_loss, _ = loss_fn(test_pred_X, test_Y)

				writer.add_scalar(
					"Test Loss", test_loss, epoch * len(train_dataloader) + i
				)

		torch.save(env_model.state_dict(), "model.pt")


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


if __name__ == "__main__":
	main()
