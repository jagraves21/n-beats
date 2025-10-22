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
		for name in ["n_blocks", "backcast", "forecast", "n_layers", "n_theta"]:
			val = getattr(self, name)
			if not isinstance(val, int) or val <= 0:
				raise ValueError(f"{name} must be a positive integer, got {val!r}.")

		for name in ["hidden_dim"]:
			val = getattr(self, name)
			if val is not None and (not isinstance(val, int) or val <= 0):
				raise ValueError(f"{name} must be a positive integer or None, got {val!r}.")

		if not isinstance(self.shared_weights, bool):
			raise ValueError(
				f"shared_weights must be a boolean (True or False), got {type(self.shared_weights).__name__}."
			)

	@abstractmethod
	def _build_block(self):
		pass

	def forward(self, X, return_intermediates=False):
		backcast_list = list()
		forecast_list = list()
		residual_list = list()
		
		forecast_sum = None
		residual = X
		for block in self.blocks:
			backcast, forecast = block(residual)

			residual = residual - backcast
			if forecast_sum is None:
				forecast_sum = torch.zeros_like(forecast)
			forecast_sum += forecast
		
			if return_intermediates:
				backcast_list.append(backcast.detach().cpu().numpy())
				forecast_list.append(forecast.detach().cpu().numpy())
				residual_list.append(residual.detach().cpu().numpy())

		if return_intermediates:
			return backcast_list, forecast_list, residual_list, residual, forecast_sum
		else:
			return residual, forecast_sum

