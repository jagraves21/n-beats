import os
import requests

from .dataset_manager import DatasetManager
from ..utils import zip_utils

class Tourism(DatasetManager):
	DATASET_URL = "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip"

	# --- Filenames ---

	def __init__(self, project_root=None):
		super().__init__("Tourism", project_root=project_root)

	# --- Dataset Download ---
	def download_dataset(self, unzip=True, force=False):
		dataset_url = "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip"
		dataset_dir = self.get_dataset_dir()
		dataset_zip = os.path.join(dataset_dir, "27-3-Athanasopoulos1.zip")
		raw_dir = os.path.join(dataset_dir, "raw")
		
		os.makedirs(dataset_dir, exist_ok=True)
		os.makedirs(raw_dir, exist_ok=True)

		if not os.path.exists(dataset_zip) or force:
			print("Downloading dataset...")
			DatasetManager.download_file(dataset_url, dataset_zip)
			print("Dataset downloaded to:", dataset_zip)
		else:
			print("Zip file already exists at:", dataset_zip)

		if unzip:
			if os.path.exists(dataset_zip):
				print(f"Extracting dataset to: {raw_dir} ...")
				zip_utils.extract_zip(dataset_zip, raw_dir, force)
			else:
				raise FileNotFoundError("Expected ZIP file was not found after download.")

	# --- Raw Data Loaders ---
	def load_yearly_in_data(self, verbose=True):
		return self.load_raw_dataframe(
			"yearly_in.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_yearly_oos_data(self, verbose=True):
		return self.load_raw_dataframe(
			"yearly_oos.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)
	
	def load_quarterly_in_data(self, verbose=True):
		return self.load_raw_dataframe(
			"quarterly_in.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_quarterly_oos_data(self, verbose=True):
		return self.load_raw_dataframe(
			"quarterly_oos.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_monthly_in_data(self, verbose=True):
		return self.load_raw_dataframe(
			"monthly_in.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_monthly_oos_data(self, verbose=True):
		return self.load_raw_dataframe(
			"monthly_oos.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)

	# -- Processed Data Loaders ---
	def load_yearly_data(self, verbose=True):
		return self.load_processed_dataframe(
			"yearly.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)
	
	def load_quarterly_data(self, verbose=True):
		return self.load_processed_dataframe(
			"quarterly.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_monthly_data(self, verbose=True):
		return self.load_processed_dataframe(
			"monthly.csv", header=0, index_col=None, nrows=None, verbose=verbose
		)

	# --- Processed Data Savers ---
	def save_yearly_data(self, df, suppress_logs=True):
		self.save_processed_dataframe(
			df, "yearly.csv", append=False, index=False, suppress_logs=False
		)

	def save_quarterly_data(self, df, suppress_logs=True):
		self.save_processed_dataframe(
			df, "quarterly.csv", append=False, index=False, suppress_logs=False
		)

	def save_monthly_data(self, df, suppress_logs=True):
		self.save_processed_dataframe(
			df, "monthly.csv", append=False, index=False, suppress_logs=False
		)

