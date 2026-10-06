# 🎙️ SHL Spoken English Grammar Scoring Engine

An automated multimodal machine learning pipeline for predicting MOS Likert Grammar Scores ($0.0 - 5.0$) from spoken audio samples (45–60 seconds in length).

---

## 📌 Approach Overview

The engine combines **Acoustic & Prosodic Features** extracted from raw audio signals with **ASR Transcriptions & Linguistic Diversity Features**:

1. **Acoustic & Speech Dynamics**:
   - **Energy & Dynamics**: RMS energy mean, std, max, min, and skewness.
   - **Silence & Pauses**: Silence ratio, pause counts (>0.3s), pause rate (pauses/min), mean & max pause durations.
   - **Spectral Clarity & Timbre**: Spectral Centroid, Bandwidth, Rolloff, Zero Crossing Rate (ZCR).
   - **MFCCs**: 20 Mel-Frequency Cepstral Coefficients + 20 Delta MFCCs + 20 Delta-Delta MFCCs.
   - **Sub-Band Energy**: Frequency power distribution across low, mid, and high spectral bands.

2. **ASR & Textual Features**:
   - OpenAI Whisper speech recognition confidence (`avg_logprob`), text compression ratio (measuring stuttering/repetition), and non-speech probability.
   - Lexical diversity (Type-Token Ratio - TTR), word count, character count, average word length, and speech rate (words per minute).

---

## 📊 Cross-Validation Performance Benchmark

Evaluated using **5-Fold Stratified Cross-Validation** (stratified on target score quantiles):

| Model | Pearson Correlation ($r$) | Validation RMSE | **Training RMSE** | Validation MAE | Validation $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Weighted Meta-Ensemble** | **0.7688** | **0.7936** | **0.3482** | **0.6375** | **0.5892** |
| 🥇 **XGBoost Regressor** | **0.7635** | **0.8003** | **0.3201** | **0.6386** | **0.5822** |
| 🥈 **Gradient Boosting** | **0.7618** | **0.8026** | **0.3031** | **0.6420** | **0.5798** |
| 🥉 **Extra Trees Regressor** | **0.7618** | **0.8046** | **0.4266** | **0.6507** | **0.5778** |
| **LightGBM Regressor** | **0.7611** | **0.8035** | **0.3755** | **0.6418** | **0.5789** |
| **CatBoost Regressor** | **0.7554** | **0.8143** | **0.5928** | **0.6661** | **0.5675** |
| **Random Forest** | **0.7523** | **0.8174** | **0.4205** | **0.6618** | **0.5642** |
| **Bayesian Ridge** | **0.7275** | **0.8497** | **0.7439** | **0.6898** | **0.5291** |
| **Ridge Regression** | **0.7236** | **0.8679** | **0.6449** | **0.6930** | **0.5087** |

---

## 📁 Repository Structure

```
├── Grammar_Scoring_Engine.ipynb   # Main documented Jupyter Notebook report
├── submission.csv                 # Test set predictions for 216 audio files
├── README.md                      # Project documentation and interview prep guide
└── .gitignore                     # Git ignore rules for dataset & binary files
```

---

## 🚀 How to Run

1. Ensure Python 3.10+ is installed with `numpy`, `pandas`, `librosa`, `soundfile`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, and `whisper`.
2. Open [`Grammar_Scoring_Engine.ipynb`](Grammar_Scoring_Engine.ipynb) in Jupyter / VS Code and execute cells sequentially.
