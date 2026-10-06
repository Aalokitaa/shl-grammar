import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import BayesianRidge, RidgeCV, ElasticNetCV, HuberRegressor, LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

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
X_tr_imp = imputer.fit_transform(X_train_raw)
X_te_imp = imputer.transform(X_test_raw)

scaler = RobustScaler()
X_tr = scaler.fit_transform(X_tr_imp)
X_te = scaler.transform(X_te_imp)

# Test feature selection K values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_k = 35
best_rmse = 999
best_r = 0

for k in [20, 30, 35, 40, 50, 60, 80, 100, len(feature_cols)]:
    oof = np.zeros(len(y_train))
    for tr, val in kf.split(X_tr, y_train):
        selector = SelectKBest(f_regression, k=min(k, X_tr.shape[1]))
        X_tr_sel = selector.fit_transform(X_tr[tr], y_train[tr])
        X_val_sel = selector.transform(X_tr[val])
        
        m = BayesianRidge(alpha_1=1e-1, alpha_2=1e-1, lambda_1=1e-1, lambda_2=1e-1)
        m.fit(X_tr_sel, y_train[tr])
        oof[val] = m.predict(X_val_sel)
    
    oof_clipped = np.clip(oof, 0.0, 5.0)
    rmse = np.sqrt(mean_squared_error(y_train, oof_clipped))
    r, _ = pearsonr(y_train, oof_clipped)
    print(f"K = {k:3d} | Val RMSE: {rmse:.4f} | Pearson r: {r:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_r = r
        best_k = k

print(f"\nOptimal K = {best_k} with CV RMSE = {best_rmse:.4f}, r = {best_r:.4f}")

# Now build 5-Fold CV predictions for test set using optimal K
oof_ensemble = np.zeros(len(y_train))
test_preds = np.zeros(len(X_te))

for fold, (tr, val) in enumerate(kf.split(X_tr, y_train)):
    selector = SelectKBest(f_regression, k=min(best_k, X_tr.shape[1]))
    X_tr_sel = selector.fit_transform(X_tr[tr], y_train[tr])
    X_val_sel = selector.transform(X_tr[val])
    X_te_sel = selector.transform(X_te)
    
    m1 = BayesianRidge(alpha_1=1e-1, alpha_2=1e-1, lambda_1=1e-1, lambda_2=1e-1)
    m2 = RidgeCV(alphas=np.logspace(-1, 4, 50))
    m3 = HuberRegressor(alpha=50.0, max_iter=1000)
    
    m1.fit(X_tr_sel, y_train[tr])
    m2.fit(X_tr_sel, y_train[tr])
    m3.fit(X_tr_sel, y_train[tr])
    
    val_pred = (m1.predict(X_val_sel) * 0.4 + m2.predict(X_val_sel) * 0.4 + m3.predict(X_val_sel) * 0.2)
    te_pred = (m1.predict(X_te_sel) * 0.4 + m2.predict(X_te_sel) * 0.4 + m3.predict(X_te_sel) * 0.2)
    
    oof_ensemble[val] = val_pred
    test_preds += te_pred / 5.0

test_preds_clipped = np.clip(test_preds, 0.0, 5.0)

sub4 = pd.DataFrame({'filename': test_df['filename'], 'label': np.round(test_preds_clipped, 4)})
sub4_path = Path(r"c:\Users\chibb\OneDrive\Desktop\SHL\submission_4_opt_ridge.csv")
sub4.to_csv(sub4_path, index=False)

print(f"\nGenerated Submission 4: {sub4_path}")
print(f"Prediction Stats: Min={test_preds_clipped.min():.4f}, Max={test_preds_clipped.max():.4f}, Mean={test_preds_clipped.mean():.4f}")
