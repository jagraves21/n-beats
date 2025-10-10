from abc import ABC, abstractmethod

import torch
import torch.nn

class NBeatsStack(torch.nn.Module, ABC):
	def __init__(self, n_blocks, backcast, forecast, n_layers, n_theta, hidden_dim, shared_weights):
		super().__init__()
		self.n_blocks = n_blocks
		self.backcast = backcast
		self.forecast = forecast
		self.n_layers = n_layers
		self.n_theta = n_theta
		self.hidden_dim = hidden_dim
		self.shared_weights = shared_weights

		self._validate_params()
		
		if self.shared_weights:
			shared_block = self._build_block()
			self.blocks = torch.nn.ModuleList([
				shared_block for _ in range(self.n_blocks)
			])
		else:
			self.blocks = torch.nn.ModuleList([
				self._build_block() for _ in range(self.n_blocks)
			])

	def _validate_params(self):
		for name in ["n_blocks", "backcast", "forecast", "n_theta"]:
			val = getattr(self, name)
			if not isinstance(val, int) or val <= 0:
				raise ValueError(f"{name} must be a positive integer, got {val!r}.")
				
		if not isinstance(self.shared_weights, bool):
			raise ValueError(
				f"shared_weights must be a boolean (True or False), got {type(self.shared_weights).__name__}."
			)

	@abstractmethod
	def _build_block(self):
		pass

	def forward(self, X):
		residual = X
		backcast_list = list()
		forecast_list = list()
		residual_list = list()

		for block in self.blocks:
			backcast, forecast = block(residual)

			backcast_list.append(backcast)
			forecast_list.append(forecast)

			residual = residual - backcast
			residual_list.append(residual)

		forecast_sum = torch.stack(forecast_list, dim=0).sum(dim=0)
		return backcast_list, forecast_list, residual_list, residual, forecast_sum
