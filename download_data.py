import os
import sys
import json
from pathlib import Path

# Ensure .kaggle directory exists
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
kaggle_json_path = kaggle_dir / "kaggle.json"

print(f"Checking Kaggle config at: {kaggle_json_path}")

if not kaggle_json_path.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
    print("\n[!] kaggle.json not found and KAGGLE environment variables are not set.")
    print("Please place kaggle.json at:", kaggle_json_path)
    print("Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
    sys.exit(1)

import kagglehub

try:
    print("Starting download for competition 'shl-hiring-assessment-2026'...")
    path = kagglehub.competition_download('shl-hiring-assessment-2026')
    print("\n[+] Dataset downloaded successfully!")
    print("Path to competition files:", path)
except Exception as e:
    print(f"\n[-] Error downloading dataset: {e}")
