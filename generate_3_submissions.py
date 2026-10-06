import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

data_dir = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\Dataset_Final")
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")

ac_file = data_dir / "acoustic_features.csv"
df_ac = pd.read_csv(ac_file).drop_duplicates(subset=['filename'])

train_merged = train_df.merge(df_ac.drop(columns=['label', 'split'], errors='ignore'), on='filename', how='left')
test_merged = test_df.merge(df_ac.drop(columns=['label', 'split'], errors='ignore'), on='filename', how='left')

ignore_cols = ['filename', 'label', 'split', 'text']
feature_cols = [c for c in train_merged.columns if c not in ignore_cols]

X_train_raw = train_merged[feature_cols].copy()
y_train = train_merged['label'].values
X_test_raw = test_merged[feature_cols].copy()

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# Model 1: Bayesian Ridge (Baseline)
m1 = BayesianRidge()
m1.fit(X_train_scaled, y_train)
p1 = np.clip(m1.predict(X_test_scaled), 0.0, 5.0)

sub1 = pd.DataFrame({'filename': test_df['filename'], 'label': np.round(p1, 4)})
sub1_path = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\submission_1_ridge.csv")
sub1.to_csv(sub1_path, index=False)
print(f"Saved Submission 1 (Baseline Ridge) to {sub1_path}")

# Model 2: XGBoost Regressor
m2 = xgb.XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
m2.fit(X_train_scaled, y_train)
p2 = np.clip(m2.predict(X_test_scaled), 0.0, 5.0)

sub2 = pd.DataFrame({'filename': test_df['filename'], 'label': np.round(p2, 4)})
sub2_path = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\submission_2_xgboost.csv")
sub2.to_csv(sub2_path, index=False)
print(f"Saved Submission 2 (Single XGBoost) to {sub2_path}")

# Model 3: Weighted Ensemble (Top 4)
m_gb = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42)
m_et = ExtraTreesRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
m_lgb = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1)

m_gb.fit(X_train_scaled, y_train)
m_et.fit(X_train_scaled, y_train)
m_lgb.fit(X_train_scaled, y_train)

p3 = np.clip((m2.predict(X_test_scaled) + m_gb.predict(X_test_scaled) + m_et.predict(X_test_scaled) + m_lgb.predict(X_test_scaled)) / 4.0, 0.0, 5.0)

sub3 = pd.DataFrame({'filename': test_df['filename'], 'label': np.round(p3, 4)})
sub3_path = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\submission.csv")
sub3.to_csv(sub3_path, index=False)
print(f"Saved Submission 3 (Final Weighted Ensemble) to {sub3_path}")
