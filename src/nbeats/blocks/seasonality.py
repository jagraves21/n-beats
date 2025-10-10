import numpy as np
import torch
import torch.nn

from .base import NBeatsBlock

class SeasonalityBlock(NBeatsBlock):
	def __init__(self, backcast, forecast, n_harmonics, n_layers=4, hidden_dim=None):
		self.n_harmonics = n_harmonics
		super().__init__(
			backcast=backcast, forecast=forecast, n_layers=n_layers,
			n_theta=2*n_harmonics+1, hidden_dim=hidden_dim)
		
		self.backcast_harmonic_basis = torch.nn.Linear(self.n_theta, self.backcast, bias=False)
		self.forecast_harmonic_basis = torch.nn.Linear(self.n_theta, self.forecast, bias=False)
		
		with torch.no_grad():
			self.backcast_harmonic_basis.weight.copy_(
				torch.from_numpy( self._get_backcast_basis() ).to(self.backcast_harmonic_basis.weight.dtype)
			)
			self.forecast_harmonic_basis.weight.copy_(
				torch.from_numpy( self._get_forecast_basis() ).to(self.forecast_harmonic_basis.weight.dtype)
			)

		self.backcast_harmonic_basis.weight.requires_grad = False
		self.forecast_harmonic_basis.weight.requires_grad = False
		
	def _validate_params(self):
		super()._validate_params()
		for name in ["n_harmonics"]:
			val = getattr(self, name)
			if not isinstance(val, int) or val < 0:
				raise ValueError(f"{name} must be a non-negative integer, got {val!r}.")

	def _get_basis(self, tt):
		harmonics = np.arange(1, self.n_harmonics + 1)
		cos_terms = np.cos(2 * np.pi * tt[:, None] * harmonics)
		sin_terms = np.sin(2 * np.pi * tt[:, None] * harmonics)
		basis = np.hstack([np.ones((len(tt), 1)), cos_terms, sin_terms])
		return basis
	
	def _get_backcast_basis(self):
		tt = np.arange(0, self.backcast) / self.backcast - 1.0
		return self._get_basis(tt)

	def _get_forecast_basis(self):
		tt = np.arange(0, self.forecast) / self.forecast
		return self._get_basis(tt)

	def get_basis_vectors(self):
		backcast_vectors = self.backcast_harmonic_basis.weight.detach().cpu().numpy()
		forecast_vectors = self.forecast_harmonic_basis.weight.detach().cpu().numpy()
		return backcast_vectors.T, forecast_vectors.T

	def forward(self, X):
		backcast_theta, forecast_theta = super().forward(X)
		backcast = self.backcast_harmonic_basis(backcast_theta)
		forecast = self.forecast_harmonic_basis(forecast_theta)
		return backcast, forecast
	
	def reset_parameters(self):
		# Do NOT reset the harmonic basis.
		super().reset_parameters()

