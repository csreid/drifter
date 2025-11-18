import torch
from tqdm import tqdm
from torch.nn import (
	Linear,
	Module,
	Sequential,
	SiLU,
)
from torch.optim import Adam
from torch.nn import functional as F
from drifter_env import observation_space, action_space
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from memory import State, Action
from batched_memory import (
	BatchedState,
	BatchedAction,
	BatchedStateDelta,
)
from torch.utils.tensorboard import SummaryWriter


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
		# "goal_position": F.mse_loss(
		# inp.local_goal_position._tensor, targets.local_goal_position._tensor
		# ),
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
		# "goal_position": 1.0,
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
		action_space, observation_space, hidden_size=2048, hidden_layers=4
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

			pred_X = BatchedStateDelta.from_tensor(env_model(state_X, action_X))
			loss, per_output_loss = loss_fn(pred_X, batch_Y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			writer.add_scalars(
				"loss_components",
				per_output_loss,
				epoch * len(train_dataloader) + i,
			)
			writer.add_scalar("Loss", loss, epoch * len(train_dataloader) + i)

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


if __name__ == "__main__":
	main()
