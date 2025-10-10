from .base import NBeatsStack
from ..blocks import TrendBlock

class TrendStack(NBeatsStack):
	def __init__(self, n_blocks, backcast, forecast, n_layers, degree, hidden_dim=None, shared_weights=False):
		self.degree = degree
		super().__init__(
			n_blocks=n_blocks,
			backcast=backcast,
			forecast=forecast,
			n_layers=n_layers,
			n_theta=degree+1,
			hidden_dim=hidden_dim,
			shared_weights=shared_weights
		)
		
	def _validate_params(self):
		super()._validate_params()
		for name in ["degree"]:
			val = getattr(self, name)
			if not isinstance(val, int) or val < 0:
				raise ValueError(f"{name} must be a non-negative integer, got {val!r}.")

	def _build_block(self):
		return TrendBlock(
			backcast=self.backcast,
			forecast=self.forecast,
			degree=self.degree,
			n_layers=self.n_layers,
			hidden_dim=self.hidden_dim
		)

