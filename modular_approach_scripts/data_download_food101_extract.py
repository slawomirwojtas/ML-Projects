"""
Contains functionality to download and unzip prepared data
for training and testing from target URL.
"""

import requests
import zipfile
from pathlib import Path

# Setup path to a data folder
data_path = Path("data/")
image_path = data_path / "food101_extract"

# Create data folder if doesn't exist
if image_path.is_dir():
  print(f"{image_path} directory already exists")
else:
  image_path.mkdir(parents=True, exist_ok=True)
  print(f"{image_path} directory created")

# Download food101_extract data
with open(data_path / "food101_extract.zip", "wb") as f:
  request = requests.get("https://github.com/slawomirwojtas/ML-Projects/raw/main/food101_extract.zip")
  print("Downloading food101_extract data...")
  f.write(request.content)

# Unzip food101_extract data
with zipfile.ZipFile(data_path / "food101_extract.zip", "r") as zip_ref:
  print("Unzipping food101_extract data...")
  zip_ref.extractall(image_path)
  print("Done")
