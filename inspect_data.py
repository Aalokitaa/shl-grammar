import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\Dataset_Final")

train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")
sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

print("="*50)
print("TRAIN DF SHAPE:", train_df.shape)
print("TRAIN DF COLUMNS:\n", train_df.columns.tolist())
print("\nTRAIN DF HEAD:")
print(train_df.head(10))

print("\n"+"="*50)
print("TEST DF SHAPE:", test_df.shape)
print("TEST DF COLUMNS:\n", test_df.columns.tolist())
print("\nTEST DF HEAD:")
print(test_df.head(10))

print("\n"+"="*50)
print("SAMPLE SUBMISSION SHAPE:", sample_sub.shape)
print("SAMPLE SUBMISSION COLUMNS:\n", sample_sub.columns.tolist())
print("\nSAMPLE SUBMISSION HEAD:")
print(sample_sub.head(10))

print("\n"+"="*50)
print("TRAIN MISSING VALUES:")
print(train_df.isnull().sum())

print("\nTRAIN DATA TYPES & INFO:")
print(train_df.info())

print("\nTRAIN SUMMARY STATS:")
print(train_df.describe(include='all'))
