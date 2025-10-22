import numpy as np
import pandas as pd

from .dataset_manager import DatasetManager
from ..utils import zip_utils

class Synthetic(DatasetManager):
	# --- Filenames ---
	TRAINING_FILE = "training.csv"
	TESTING_FILE = "testing.csv"

	def __init__(self, project_root=None):
		super().__init__("Synthetic", project_root=project_root)

	# --- Dataset Download ---
	def download_dataset(self, unzip=True, force=False):
		self.log("Generating synthetic data...")
		df = Synthetic.generate_data(
			n_periods=20,
			n_samples_per_period=10,
			n_harmonics=2,
			loc=0,
			scale=0.5,
			seed=42
		)
		
		split_idx = int(len(df) * 0.9)
		training_df = df.iloc[:split_idx]
		testing_df = df.iloc[split_idx:]
		
		path = self.get_raw_file_path(self.TRAINING_FILE)
		self._save_csv_dataframe(
			training_df, path, append=False, index=False, suppress_logs=False,
			log_message=f"Saved DataFrame to processed directory: {path}"
		)

		path = self.get_raw_file_path(self.TESTING_FILE)
		self._save_csv_dataframe(
			testing_df, path, append=False, index=False, suppress_logs=False,
			log_message=f"Saved DataFrame to processed directory: {path}"
		)



	# --- Raw Data Loaders ---
	def load_training_data(self, verbose=True):
		return self.load_raw_dataframe(
			self.TRAINING_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_testing_data(self, verbose=True):
		return self.load_raw_dataframe(
			self.self.TESTING_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	# --- Data Generation --- 
	@staticmethod
	def generate_data(n_periods=20, n_samples_per_period=10, n_harmonics=2, loc=0, scale=0.5, seed=42):
		random = np.random.default_rng(seed=seed)
		
		tt = np.linspace(0, 2 * n_periods * np.pi, n_samples_per_period * n_periods)
		seasonal_components = np.column_stack([
			func(hh * tt) / hh
			for hh in range(1, n_harmonics + 1)
			for func in (np.sin, np.cos)
		])
		trend = 0.001 * tt**2 - 0.001 * tt + 5
		seasonal = seasonal_components.sum(axis=1)
		noise = random.normal(loc=0, scale=0.5, size=trend.shape)
		
		series = trend + seasonal + noise
		
		data = dict(
			series=series,
			trend=trend,
			seasonal=seasonal,
			noise=noise
		)
		for ii in range(seasonal_components.shape[1]):
			data[f"component_{ii+1}"] = seasonal_components[:, ii]
		
		return pd.DataFrame(data)

