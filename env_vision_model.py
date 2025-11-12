import torch
from tqdm import tqdm
from collections import deque
from torch.nn import (
	Sequential,
	Linear,
	Module,
	Sequential,
	LeakyReLU,
	MSELoss,
	Sigmoid,
	SiLU,
	Conv2d,
	MaxPool2d,
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


def _get_output_shape(model, input_shape):
	"""
	Calculate the output shape of a PyTorch Sequential model.

	Args:
			model: torch.nn.Sequential module
			input_shape: tuple of (channels, height, width) or (batch, channels, height, width)

	Returns:
			tuple: output shape (channels, height, width)
	"""
	# Ensure input_shape has batch dimension
	if len(input_shape) == 3:
		input_shape = (1,) + input_shape

	# Create a dummy input tensor
	dummy_input = torch.randn(*input_shape)

	# Forward pass through the model
	with torch.no_grad():
		output = model(dummy_input)

	# Return shape without batch dimension
	return tuple(output.shape[1:])


class EnvModel(Module):
	def __init__(self):
		super().__init__()
		self._viz_pipeline = Sequential(
			Conv2d(3, 16, kernel_size=4, stride=2),
			MaxPool2d(kernel_size=4, stride=2),
			LeakyReLU(),

			Conv2d(16, 64, kernel_size=3),
			MaxPool2d(kernel_size=3),
			LeakyReLU(),

			Conv2d(64, 128, kernel_size=3),
			MaxPool2d(kernel_size=3),
			LeakyReLU(),

			Conv2d(128, 512, kernel_size=3),
			LeakyReLU(),
		)

		out_shape = _get_output_shape(self._viz_pipeline, (3,  160, 240))
		self.h1 = Linear(out_shape, 512)

		self.out = Linear(512, 16)

	def forward(self, X):
		out = self._viz_pipeline(X)
		out = self.h1(out)
		out = F.leaky_relu(out)
		out = self.out(out)
