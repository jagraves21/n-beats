import numpy as np
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
	def __init__(
		self,
		XX,
		yy=None,
		backcast=10,
		forecast=5,
		include_backcast_in_y=False,
		to_tensor=True,
	):
		super().__init__()
		self.XX = XX
		self.yy = yy
		self.backcast = backcast
		self.forecast = forecast
		self.include_backcast_in_y = include_backcast_in_y
		self.to_tensor = to_tensor
		
		self._validate_params()

		self.XX = np.asarray(self.XX)
		if self.XX.ndim == 1:
			self.XX = self.XX.reshape(-1, 1)

		if self.yy is None:
			self.yy = self.XX
		else:
			self.yy = np.asarray(self.yy)
			if self.yy.ndim == 1:
				self.yy = self.yy.reshape(-1, 1)
			if len(self.yy) != len(self.XX):
				raise ValueError(f"XX and yy must have the same length, got {len(self.XX)} and {len(self.yy)}.")
		
		if self.to_tensor:
			self.XX = torch.from_numpy(self.XX.astype(np.float32))
			self.yy = torch.from_numpy(self.yy.astype(np.float32))

	def _validate_params(self):
		for name in ["backcast", "forecast"]:
			val = getattr(self, name)
			if not isinstance(val, int) or val <= 0:
				raise ValueError(f"{name} must be a positive integer, got {val!r}.")
		
		for name in ["include_backcast_in_y", "to_tensor"]:
			val = getattr(self, name)
			if not isinstance(val, bool):
				raise ValueError(f"{name} must be a boolean (True or False), got {type(val).__name__}.")
		
		if not isinstance(self.XX, (np.ndarray, torch.Tensor)):
			raise TypeError(f"XX must be a numpy array or torch tensor, got {type(self.XX).__name__}.")

		if self.yy is not None and not isinstance(self.yy, (np.ndarray, torch.Tensor)):
			raise TypeError(f"yy must be a numpy array or torch tensor, got {type(self.yy).__name__}.")

		XX_len = len(self.XX)
		if XX_len < self.backcast + self.forecast:
			raise ValueError(
				f"Length of XX ({XX_len}) must be at least backcast + forecast ({self.backcast + self.forecast})."
			)

	def __len__(self):
		return max(0, len(self.XX) - (self.backcast + self.forecast) + 1)

	def __getitem__(self, idx):
		if isinstance(idx, slice):
			return self._getslice(idx)
		else:
			return self._getitem(idx)

	def _getitem(self, idx):
		n = len(self)
		if idx < 0:
			idx += n
		if idx < 0 or idx >= n:
			raise IndexError("index out of range")

		xx = self.XX[idx : idx + self.backcast]
		if xx.ndim == 1:
			xx = xx.reshape(-1, 1)

		start_y = idx if self.include_backcast_in_y else idx + self.backcast
		end_y = idx + self.backcast + self.forecast
		yy = self.yy[start_y:end_y]
		if yy.ndim == 1:
			yy = yy.reshape(-1, 1)

		return xx, yy

	def _getslice(self, idx):
		indices = range(*idx.indices(len(self)))
		xx_list, yy_list = [], []
		for ii in indices:
			tmp_x, tmp_y = self._getitem(ii)
			xx_list.append(tmp_x)
			yy_list.append(tmp_y)
		if not xx_list:
			if self.to_tensor:
				return torch.empty(0), torch.empty(0)
			else:
				return np.empty((0, 0), dtype=self.XX.dtype), np.empty((0, 0), dtype=self.yy.dtype)
		if self.to_tensor:
			return torch.stack(xx_list), torch.stack(yy_list)
		else:
			return np.stack(xx_list), np.stack(yy_list)

