# https://research.monash.edu/en/publications/the-tourism-forecasting-competition
# https://robjhyndman.com/publications/the-tourism-forecasting-competition/

import os
import requests

from .dataset_manager import DatasetManager
from ..utils import zip_utils

class Tourism(DatasetManager):
	DATASET_URL = "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip"

	# --- Filenames ---
	YEARLY_IN_FILE = "yearly_in.csv"
	YEARLY_OOS_FILE = "yearly_oos.csv"
	QUARTERLY_IN_FILE = "quarterly_in.csv"
	QUARTERLY_OOS_FILE = "quarterly_oos.csv"
	MONTHLY_IN_FILE = "monthly_in.csv"
	MONTHLY_OOS_FILE = "monthly_oos.csv"

	YEARLY_FILE = "yearly.csv"
	QUARTERLY_FILE = "quarterly.csv"
	MONTHLY_FILE = "monthly.csv"

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
			YEARLY_IN_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_yearly_oos_data(self, verbose=True):
		return self.load_raw_dataframe(
			YEARLY_OOS_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)
	
	def load_quarterly_in_data(self, verbose=True):
		return self.load_raw_dataframe(
			QUARTERLY_IN_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_quarterly_oos_data(self, verbose=True):
		return self.load_raw_dataframe(
			QUARTERLY_OOS_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_monthly_in_data(self, verbose=True):
		return self.load_raw_dataframe(
			MONTHLY_IN_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_monthly_oos_data(self, verbose=True):
		return self.load_raw_dataframe(
			MONTHLY_OOS_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	# -- Processed Data Loaders ---
	def load_yearly_data(self, verbose=True):
		return self.load_processed_dataframe(
			YEARLY_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)
	
	def load_quarterly_data(self, verbose=True):
		return self.load_processed_dataframe(
			QUARTERLY_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_monthly_data(self, verbose=True):
		return self.load_processed_dataframe(
			MONTHLY_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	# --- Processed Data Savers ---
	def save_yearly_data(self, df, suppress_logs=True):
		self.save_processed_dataframe(
			df, YEARLY_FILE, append=False, index=False, suppress_logs=False
		)

	def save_quarterly_data(self, df, suppress_logs=True):
		self.save_processed_dataframe(
			df, QUARTERLY_FILE, append=False, index=False, suppress_logs=False
		)

	def save_monthly_data(self, df, suppress_logs=True):
		self.save_processed_dataframe(
			df, MONTHLY_FILE, append=False, index=False, suppress_logs=False
		)

