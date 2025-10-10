from .base import NBeatsStack
from ..blocks import GenericBlock

class GenericStack(NBeatsStack):
	def __init__(self, n_blocks, backcast, forecast, n_layers, n_theta, hidden_dim=None, shared_weights=False):
		super().__init__(
			n_blocks=n_blocks,
			backcast=backcast,
			forecast=forecast,
			n_layers=n_layers,
			n_theta=n_theta,
			hidden_dim=hidden_dim,
			shared_weights=shared_weights
		)

	def _build_block(self):
		return GenericBlock(
			backcast=self.backcast,
			forecast=self.forecast,
			n_theta=self.n_theta,
			n_layers=self.n_layers,
			hidden_dim=self.hidden_dim
		)

