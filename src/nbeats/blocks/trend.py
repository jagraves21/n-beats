import warnings

import numpy as np
import torch
import torch.nn

from .base import NBeatsBlock

class TrendBlock(NBeatsBlock):
	def __init__(self, backcast, forecast, degree, n_layers=4, hidden_dim=None):
		self.degree = degree
		super().__init__(
			backcast=backcast, forecast=forecast, n_layers=n_layers,
			n_theta=degree+1, hidden_dim=hidden_dim
		)
		
		self.backcast_polynomial_basis = torch.nn.Linear(self.n_theta, self.backcast, bias=False)
		self.forecast_polynomial_basis = torch.nn.Linear(self.n_theta, self.forecast, bias=False)
		
		with torch.no_grad():
			self.backcast_polynomial_basis.weight.copy_(
				torch.from_numpy( self._get_backcast_basis() ).to(
					self.backcast_polynomial_basis.weight.dtype
				)
			)
			self.forecast_polynomial_basis.weight.copy_(
				torch.from_numpy( self._get_forecast_basis() ).to(
					self.forecast_polynomial_basis.weight.dtype
				)
			)

		self.backcast_polynomial_basis.weight.requires_grad = False
		self.forecast_polynomial_basis.weight.requires_grad = False
		
	def _validate_params(self):
		super()._validate_params()
		for name in ["degree"]:
			val = getattr(self, name)
			if not isinstance(val, int) or val < 0:
				raise ValueError(f"{name} must be a non-negative integer, got {val!r}.")

		if self.degree > min(self.backcast, self.forecast) // 2:
			warnings.warn(
				f"degree={self.degree} may be too high for backcast={self.backcast} "
				f"and forecast={self.forecast}. Consider using a lower degree "
				f"(<= {min(self.backcast, self.forecast) // 2}) to avoid numerical instability or overfitting.",
				UserWarning,
				stacklevel=2,
			)

	def _get_basis(self, tt):
		basis = np.vander(tt, N=self.n_theta, increasing=True)
		return basis
	
	def _get_backcast_basis(self):
		tt = np.arange(0, self.backcast) / self.backcast - 1.0
		return self._get_basis(tt)

	def _get_forecast_basis(self):
		tt = np.arange(0, self.forecast) / self.forecast
		return self._get_basis(tt)

	def get_basis_vectors(self):
		backcast_vectors = self.backcast_polynomial_basis.weight.detach().cpu().numpy()
		forecast_vectors = self.forecast_polynomial_basis.weight.detach().cpu().numpy()
		return backcast_vectors.T, forecast_vectors.T

	def forward(self, X):
		backcast_theta, forecast_theta = super().forward(X)
		backcast = self.backcast_polynomial_basis(backcast_theta)
		forecast = self.forecast_polynomial_basis(forecast_theta)
		return backcast, forecast
	
	def reset_parameters(self):
		# Do NOT reset the polynomial basis.
		super().reset_parameters()

