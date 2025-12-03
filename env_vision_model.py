import torch
from torch.nn import (
	Linear,
	Module,
	Sequential,
	LeakyReLU,
	Conv2d,
	MaxPool2d,
	AdaptiveAvgPool2d,
	Flatten,
	LSTM,
	ModuleList,
	ModuleDict
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
	def __init__(self, hidden_size=512):
		super().__init__()
		self._viz_pipeline = Sequential(
			Conv2d(3, 16, kernel_size=4, stride=2),
			LeakyReLU(),
			Conv2d(16, 64, kernel_size=3, stride=2),
			LeakyReLU(),
			Conv2d(64, 128, kernel_size=3, stride=2),
			LeakyReLU(),
			Conv2d(128, 512, kernel_size=3, stride=2),
			AdaptiveAvgPool2d((1, 1)),
			LeakyReLU(),
			Flatten(),
		)

		viz_out_shape = reduce(
			lambda acc, val: acc * val,
			_get_output_shape(self._viz_pipeline, (3, 480, 480)),
		)

		self._h1 = Linear(viz_out_shape, hidden_size)

		self._rnn = LSTM(hidden_size, hidden_size, batch_first=True)

		self._dynamics_output_heads = ModuleDict({
			'velocity': Linear(hidden_size, 3),
			'position': Linear(hidden_size, 3),
			'orientation': Linear(hidden_size, 4),
			'goal_position': Linear(hidden_size, 3),
			'local_goal_position': Linear(hidden_size, 3)
		})

		self._id_output_head = Linear(hidden_size, 2) # for 2-d action; steering and throttle/brake/reverse

	def _get_hidden(self, imgs, seqlens):
		batchsize, seqlen, C, H, W = imgs.shape

		out = imgs.view(seqlen * batchsize, C, H, W)
		out = self._viz_pipeline(out)
		out = self._h1(out)
		out = F.leaky_relu(out)

		embed_dim = out.shape[-1]
		out = out.view(batchsize, seqlen, embed_dim)

		out = pack_padded_sequence(
			out, seqlens, batch_first=True, enforce_sorted=False
		)
		out, _ = self._rnn(out)
		out, lens_out = pad_packed_sequence(out, batch_first=True)
		out = F.leaky_relu(out)

		return out

	def inverse_dynamics(self, imgs, seqlens):
		out = self._get_hidden(imgs, seqlens)
		out = self._id_output_head(out)

		return out

	def forward(self, imgs, seqlens):
		out = self._get_hidden(imgs, seqlens)

		return {
			key: val(out)
			for key, val
			in self._dynamics_output_heads.items()
		}
