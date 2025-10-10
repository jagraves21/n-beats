from abc import ABC, abstractmethod

import torch
import torch.nn

class NBeatsBlock(torch.nn.Module, ABC):
	def __init__(self, backcast, forecast, n_layers, n_theta, hidden_dim):
		super().__init__()
		self.backcast = backcast
		self.forecast = forecast
		self.n_theta = n_theta
		self.n_layers = n_layers
		self.hidden_dim = hidden_dim or backcast
		
		self._validate_params()
		
		if self.n_layers <= 0:
			raise ValueError("n_layers must be greater than zero.")

		self.fc_stack = self._build_fc_stack()
		self.theta = torch.nn.Linear(self.hidden_dim, 2 * self.n_theta, bias=False)
		
	def _validate_params(self):
		for name in ["backcast", "forecast", "n_theta", "n_layers", "hidden_dim"]:
			val = getattr(self, name)
			if not isinstance(val, int) or val <= 0:
				raise ValueError(f"{name} must be a positive integer, got {val!r}.")

	def _build_fc_stack(self):
		layers = list()
		for ii in range(self.n_layers):
			in_features = self.backcast if ii == 0 else self.hidden_dim
			out_features = self.hidden_dim
			layers.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
			layers.append(torch.nn.ReLU(inplace=False))
		return torch.nn.Sequential(*layers)

	@abstractmethod
	def get_basis_vectors(self):
		pass

	def forward(self, X):
		theta = self.theta( self.fc_stack(X) )
		backcast_theta, forecast_theta = torch.split(theta, self.n_theta, dim=-1)
		return backcast_theta, forecast_theta

	def reset_parameters(self):
		for layer in self.fc_stack:
			if isinstance(layer, torch.nn.Linear):
				layer.reset_parameters()
		self.theta.reset_parameters()

