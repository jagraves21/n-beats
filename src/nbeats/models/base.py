from abc import ABC, abstractmethod

import torch.nn

class NBeatsModelBase(torch.nn.Module, ABC):
	def __init__(self):
		super().__init__()
		self.stacks = torch.nn.ModuleList( self._build_stacks() )

	@abstractmethod
	def _build_stacks(self):
		pass

	def forward(self, X):
		residual = X
		all_backcasts = []
		all_forecasts = []
		all_residuals = []
		stack_forecasts = []

		for stack in self.stacks:
			backcast_list, forecast_blocks, residual_list, residual, forecast = stack(residual)
			all_backcasts.append(backcast_list)
			all_forecasts.append(forecast_blocks)
			all_residuals.append(residual_list)
			stack_forecasts.append(forecast)

		forecast_sum = sum(stack_forecasts)
		return all_backcasts, all_forecasts, all_residuals, forecast_sum

