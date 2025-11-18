import torch
from torch.nn import (
	Linear,
	Module,
	Sequential,
	LeakyReLU,
	Conv2d,
	MaxPool2d,
	Flatten,
	LSTM,
)
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import reduce


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
			Flatten(),
		)

		viz_out_shape = reduce(
			lambda acc, val: acc * val,
			_get_output_shape(self._viz_pipeline, (3, 480, 480)),
		)

		self._h1 = Linear(viz_out_shape, 512)

		self._rnn = LSTM(512, 512, batch_first=True)

		self.velocity_head = Linear(512, 3)
		self.position_head = Linear(512, 3)
		self.orientation_head = Linear(512, 4)
		self.goal_position_head = Linear(512, 3)
		self.local_goal_position_head = Linear(512, 3)

	def forward(self, X, seqlens):
		batchsize, seqlen, C, H, W = X.shape

		out = X.view(seqlen * batchsize, C, H, W)
		out = self._viz_pipeline(out)
		out = self._h1(out)
		out = F.leaky_relu(out)

		embed_dim = out.shape[-1]
		out = out.view(batchsize, seqlen, embed_dim)

		out = pack_padded_sequence(out, seqlens, batch_first=True, enforce_sorted=False)
		out, _= self._rnn(out)
		out, lens_out = pad_packed_sequence(out, batch_first=True)
		out = F.leaky_relu(out)

		velocity_out = self.velocity_head(out)
		position_out = self.position_head(out)
		orientation_out = self.orientation_head(out)
		#goal_position_out = self.goal_position_head(out)
		#local_goal_position_out = self.local_goal_position_head(out)

		return {
			"position": position_out,
			"velocity": velocity_out,
			"orientation": orientation_out,
			#"goal": goal_position_out,
			#"local_goal": local_goal_position_out,
		}
