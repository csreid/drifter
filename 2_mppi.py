import torch
from drifter_env import DrifterEnv
from pytorch_mppi import MPPI
from env_model import EnvModel
from gymnasium.action_spaces import Box

N_SAMPLES = 1000
TIMESTEPS = 20
lambda_ = torch.tensor(1.0).float()
d = "cpu"

env = DrifterEnv(gui=True)
env_model = EnvModel(
	env.action_space,
	env.observation_space,
	hidden_size=1024,
	hidden_layers=3,
)
env_model.load_state_dict(torch.load("model.pt", weights_only=True))
env_model.eval()


def compute_cost(state, action):
	local_goal_pos = state[..., :3]
	is_flipped = torch.sigmoid(state[..., 6])

	# Distance to goal at this timestep
	distance = torch.norm(local_goal_pos, dim=-1)

	# Costs at this timestep
	flipped_cost = 1000.0 * is_flipped
	action_cost = 0.01 * (action**2).sum(dim=-1)

	return distance + action_cost + flipped_cost  # shape: (K,) or (M, K)


assert type(env.action_space) is Box
assert type(env.action_space) is Box
ctrl = MPPI(
	env_model,
	compute_cost,
	14,
	noise_sigma=torch.eye(2, dtype=torch.float).float(),
	noise_mu=torch.zeros(2).float(),
	num_samples=N_SAMPLES,
	horizon=TIMESTEPS,
	lambda_=lambda_,
	device=d,
	u_min=torch.tensor(env.action_space.high, dtype=torch.float, device=d),
	u_max=torch.tensor(env.action_space.low, dtype=torch.float, device=d),
)

with torch.no_grad():
	# assuming you have a gym-like env
	obs, _ = env.reset()
	for i in range(100):
		action = ctrl.command(obs)
		obs, reward, done, trunc, _ = env.step(action.cpu().detach().numpy())
