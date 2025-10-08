from stable_baselines3 import DQN, PPO, DDPG

# from stander_env import QuadrupedStandEnv
from drifter_env import DrifterEnv


def train(env, steps):
	model = PPO(
		"MultiInputPolicy", env, verbose=0, tensorboard_log="logs", n_steps=512
	)
	model.learn(total_timesteps=steps, progress_bar=True, tb_log_name="drifter")


if __name__ == "__main__":
	env = DrifterEnv(gui=True)
	train(env, 5000000)
