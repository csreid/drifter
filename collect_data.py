from tqdm import tqdm
import torch
from drifter_env import DrifterEnv
from torch.utils.tensorboard import SummaryWriter
from exploration_policy import ExplorationPolicy
from memory import MPPIMemory, Transition, Action, State, StateDelta
from batched_memory import (
	BatchedTransition,
	BatchedAction,
	BatchedState,
	BatchedStateDelta,
)

def main():
	memory = MPPIMemory(100_000)
	env = DrifterEnv(gui=True)
	expl_policy = ExplorationPolicy(
		env.action_space,
		maneuver_duration=15,
		aggressive_prob=0.35,
		velocity_threshold=0.3,
	)

	s, _ = env.reset()
	done = False
	new_episode = False
	i = 0
	for i in tqdm(range(100_000)):
		a = expl_policy.get_action(s)
		# a, _ = mppi_controller.get_action(s)
		sp, r, done, trunc, _ = env.step(a)

		transition = Transition(
			action=Action.from_tensor(torch.tensor(a)),
			state=State.from_tensor(torch.tensor(s)),
			next_state=State.from_tensor(torch.tensor(sp)),
		)
		memory.add(transition)

		s = sp

		if done or trunc:
			s, _ = env.reset()

	memory.to_pandas().to_csv("transitions.csv", index=False)

if __name__ == '__main__':
	main()
