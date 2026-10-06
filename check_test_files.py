import pandas as pd
from pathlib import Path

data_dir = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\Dataset_Final")

train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")
sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

train_audio_dir = data_dir / "train"
test_audio_dir = data_dir / "test"

train_files_on_disk = set(f.name for f in train_audio_dir.glob("*.wav"))
test_files_on_disk = set(f.name for f in test_audio_dir.glob("*.wav"))

print(f"Train CSV count: {len(train_df)}")
print(f"Train files on disk in 'train/': {len(train_files_on_disk)}")
print(f"Train files in CSV missing from disk: {set(train_df['filename']) - train_files_on_disk}")

print(f"\nTest CSV count: {len(test_df)}")
print(f"Sample Submission count: {len(sample_sub)}")
print(f"Test files on disk in 'test/': {len(test_files_on_disk)}")

sample_sub_files = set(sample_sub['filename'])
test_csv_files = set(test_df['filename'])

print(f"Sample sub files in test disk: {len(sample_sub_files & test_files_on_disk)} / {len(sample_sub_files)}")
print(f"Test CSV files in test disk: {len(test_csv_files & test_files_on_disk)} / {len(test_csv_files)}")
print(f"Sample sub vs Test CSV diff: {len(sample_sub_files ^ test_csv_files)}")

# Check audio libraries
import importlib
for pkg in ["librosa", "soundfile", "scipy", "torchaudio", "whisper", "transformers", "torch"]:
    try:
        mod = importlib.import_module(pkg)
        print(f"[+] Package '{pkg}' is available (version {getattr(mod, '__version__', 'N/A')})")
    except ImportError:
        print(f"[-] Package '{pkg}' is NOT installed")
