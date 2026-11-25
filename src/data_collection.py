
import requests
import gzip
import shutil
from pathlib import Path

# Correct UCI dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"

# Get project root directory
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data" / "raw"

data_folder.mkdir(parents=True, exist_ok=True)

compressed_file = data_folder / "traffic.csv.gz"
csv_file = data_folder / "traffic.csv"

print("Downloading traffic dataset...")

response = requests.get(url, timeout=60)
response.raise_for_status()

with open(compressed_file, "wb") as file:
    file.write(response.content)

print("Download successful!")
print("Extracting dataset...")

with gzip.open(compressed_file, "rb") as source:
    with open(csv_file, "wb") as destination:
        shutil.copyfileobj(source, destination)

print("Dataset downloaded and extracted successfully!")
print("Saved at:", csv_file)
