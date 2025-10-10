import torch.nn

from .base import NBeatsBlock

class GenericBlock(NBeatsBlock):
	def __init__(self, backcast, forecast, n_theta, n_layers=4, hidden_dim=None):
		super().__init__(
			backcast=backcast, forecast=forecast, n_layers=n_layers,
			n_theta=n_theta, hidden_dim=hidden_dim
		)
		
		self.backcast_learnable_basis = torch.nn.Linear(self.n_theta, self.backcast, bias=False)
		self.forecast_learnable_basis = torch.nn.Linear(self.n_theta, self.forecast, bias=False)

	def get_basis_vectors(self):
		backcast_vectors = self.backcast_learnable_basis.weight.detach().cpu().numpy()
		forecast_vectors = self.forecast_learnable_basis.weight.detach().cpu().numpy()
		return backcast_vectors.T, forecast_vectors.T

	def forward(self, X):
		backcast_theta, forecast_theta = super().forward(X)
		backcast = self.backcast_learnable_basis(backcast_theta)
		forecast = self.forecast_learnable_basis(forecast_theta)
		return backcast, forecast
	
	def reset_parameters(self):
		super().reset_parameters()
		self.backcast_learnable_basis.reset_parameters()
		self.forecast_learnable_basis.reset_parameters()

