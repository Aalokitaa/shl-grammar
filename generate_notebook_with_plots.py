import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

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

bins = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Ridge Regression": Ridge(alpha=10.0),
    "Bayesian Ridge": BayesianRidge(),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0, n_jobs=-1),
    "CatBoost": cb.CatBoostRegressor(iterations=150, learning_rate=0.05, depth=4, verbose=0, random_seed=42, thread_count=-1)
}

results = []
oof_preds = {}
test_preds = {}

for name, model in models.items():
    oof = np.zeros(len(y_train))
    t_preds = np.zeros(len(test_df))
    train_rmses = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, bins)):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_va, y_va = X_train_scaled[val_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_va)
        t_preds += model.predict(X_test_scaled) / 5.0
        train_rmses.append(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))))
        
    oof_preds[name] = oof
    test_preds[name] = t_preds
    
    val_rmse = np.sqrt(mean_squared_error(y_train, oof))
    val_mae = mean_absolute_error(y_train, oof)
    val_r2 = r2_score(y_train, oof)
    p_corr, _ = pearsonr(y_train, oof)
    avg_tr_rmse = float(np.mean(train_rmses))
    
    results.append({
        "Model": name,
        "Pearson Corr (r)": p_corr,
        "Val RMSE": val_rmse,
        "Train RMSE": avg_tr_rmse,
        "Val MAE": val_mae,
        "Val R2": val_r2
    })

top_models = ["XGBoost", "Gradient Boosting", "Extra Trees", "LightGBM"]
stacked_oof = np.mean([oof_preds[m] for m in top_models], axis=0)
stacked_test = np.mean([test_preds[m] for m in top_models], axis=0)

ens_val_rmse = np.sqrt(mean_squared_error(y_train, stacked_oof))
ens_val_mae = mean_absolute_error(y_train, stacked_oof)
ens_val_r2 = r2_score(y_train, stacked_oof)
ens_p_corr, _ = pearsonr(y_train, stacked_oof)

ens_tr_rmses = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, bins)):
    tr_preds_fold = [models[m].fit(X_train_scaled[train_idx], y_train[train_idx]).predict(X_train_scaled[train_idx]) for m in top_models]
    ens_tr_rmses.append(np.sqrt(mean_squared_error(y_train[train_idx], np.mean(tr_preds_fold, axis=0))))

results.append({
    "Model": "Weighted Meta-Ensemble (Top 4)",
    "Pearson Corr (r)": ens_p_corr,
    "Val RMSE": ens_val_rmse,
    "Train RMSE": float(np.mean(ens_tr_rmses)),
    "Val MAE": ens_val_mae,
    "Val R2": ens_val_r2
})

results_df = pd.DataFrame(results).sort_values(by="Pearson Corr (r)", ascending=False)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

# Generate Plot 1: EDA
sns.set_theme(style="whitegrid", palette="muted")
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(train_df['label'], kde=True, ax=axes1[0], color='#2b5c8f', bins=11)
axes1[0].set_title("Distribution of Training Grammar Scores (Likert 0-5)", fontsize=13, fontweight='bold')
axes1[0].set_xlabel("Grammar Score (MOS)", fontsize=11)
axes1[0].set_ylabel("Sample Count", fontsize=11)

sns.boxplot(x=train_df['label'], ax=axes1[1], color='#4ea597')
axes1[1].set_title("Grammar Score Boxplot & Quantile Spread", fontsize=13, fontweight='bold')
axes1[1].set_xlabel("Grammar Score (MOS)", fontsize=11)
plt.tight_layout()
b64_img1 = fig_to_base64(fig1)

# Generate Plot 2: Model Comparison Chart
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(x="Pearson Corr (r)", y="Model", data=results_df, palette="Blues_r", ax=ax2)
ax2.set_title("Model Comparison - Pearson Correlation Coefficient (r)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Pearson Correlation (r)", fontsize=12)
ax2.set_xlim(0.4, 0.85)

for p in ax2.patches:
    w = p.get_width()
    ax2.annotate(f"{w:.4f}", (w + 0.005, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=10, color='#333333')

plt.tight_layout()
b64_img2 = fig_to_base64(fig2)

# Generate Plot 3: Feature Importance & Scatter
rf_m = models["Extra Trees"]
rf_m.fit(X_train_scaled, y_train)
feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_m.feature_importances_}).sort_values(by='Importance', ascending=False)

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(15), palette='viridis', ax=axes3[0])
axes3[0].set_title("Top 15 Most Important Features for Grammar Scoring", fontsize=13, fontweight='bold')
axes3[0].set_xlabel("Feature Importance Weight", fontsize=11)

axes3[1].scatter(y_train, stacked_oof, alpha=0.6, color='#1f77b4', edgecolors='k', linewidth=0.5)
axes3[1].plot([0, 5], [0, 5], 'r--', label='Perfect Prediction Diagonal')
axes3[1].set_title(f"OOF Predictions vs Ground Truth (Pearson r = {ens_p_corr:.4f})", fontsize=13, fontweight='bold')
axes3[1].set_xlabel("Ground Truth MOS Score", fontsize=11)
axes3[1].set_ylabel("Model Predicted MOS Score", fontsize=11)
axes3[1].legend()

plt.tight_layout()
b64_img3 = fig_to_base64(fig3)

print("Generated all base64 PNG visualizations successfully!")
