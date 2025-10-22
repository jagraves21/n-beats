from abc import ABC, abstractmethod

import numpy as np
import torch.nn

class NBeatsModelBase(torch.nn.Module, ABC):
	def __init__(self):
		super().__init__()
		self.stacks = torch.nn.ModuleList( self._build_stacks() )

	@abstractmethod
	def _build_stacks(self):
		pass

	def forward(self, X, return_intermediates=False):
		all_backcasts = []
		all_forecasts = []
		all_residuals = []
		stack_forecasts = []

		forecast_sum = None
		residual = X
		for stack in self.stacks:
			res = stack(residual, return_intermediates=return_intermediates)
			if return_intermediates:
				backcast_list, forecast_list, residual_list, residual, forecast = res
				all_backcasts.append(backcast_list)
				all_forecasts.append(forecast_list)
				all_residuals.append(residual_list)
				stack_forecasts.append(forecast.detach().cpu().numpy())
			else:
				residual, forecast = res
			if forecast_sum is None:
				forecast_sum = torch.zeros_like(forecast)
			forecast_sum += forecast

		if return_intermediates:
			all_backcasts = np.moveaxis(np.asarray(all_backcasts), 2, 0)
			all_forecasts = np.moveaxis(np.asarray(all_forecasts), 2, 0)
			all_residuals = np.moveaxis(np.asarray(all_residuals), 2, 0)
			return all_backcasts, all_forecasts, all_residuals, forecast_sum
		else:
			return forecast_sum

