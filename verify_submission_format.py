import pandas as pd
from pathlib import Path

data_dir = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\Dataset_Final")

test_df = pd.read_csv(data_dir / "test.csv")
sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

print("test.csv filenames head:")
print(test_df.head(10))

print("\nsample_submission.csv filenames head:")
print(sample_sub.head(10))

test_set = set(test_df['filename'])
sub_set = set(sample_sub['filename'])

print(f"\ntest.csv len: {len(test_df)}")
print(f"sample_submission.csv len: {len(sample_sub)}")

print(f"Intersection len: {len(test_set & sub_set)}")
print(f"In test.csv but not sample_sub: {len(test_set - sub_set)}")
print(f"In sample_sub but not test.csv: {len(sub_set - test_set)}")
