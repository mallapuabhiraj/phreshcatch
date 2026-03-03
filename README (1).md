# 🔍 Phishing URL Detection — ML Pipeline

> A production-ready machine learning pipeline that detects phishing URLs in real time using only URL structure — no HTML scraping, no page visits required.

---

## 📌 Project Overview

Phishing attacks remain one of the most prevalent cybersecurity threats. This project builds an end-to-end ML pipeline that classifies URLs as **phishing** or **benign** using 21 features extracted purely from the URL string itself.

The model runs at **0.4ms per URL** and achieves **92%+ accuracy** on the PhreshPhish dataset — a production-grade benchmark collected from real browsing traffic of 6 million users between July 2024 and March 2025.

---

## 🏆 Key Results

| Model | Accuracy | ROC-AUC | FN (Missed Phishing) | Train Time |
|---|---|---|---|---|
| **XGBoost** *(best)* | **92.01%** | **0.9697** | **3,903** | 54.6s |
| LightGBM | 91.90% | 0.9680 | 4,029 | 47.7s |
| Random Forest | 90.84% | 0.9646 | 3,964 | 847.2s |
| Logistic Regression | 85.53% | 0.9244 | 11,696 | 55.2s |
| Decision Tree | 86.69% | 0.8671 | 7,021 | 43.1s |

> **FN (False Negatives)** = phishing URLs the model missed. In security contexts this is the most critical metric.

---

## 🔬 Research Findings

This project uncovered three important findings that distinguish it from standard phishing detection tutorials:

### Finding 1 — URLSimilarityIndex Leakage (PhiUSIIL Dataset)
During initial experimentation with the PhiUSIIL dataset, `URLSimilarityIndex` was found to equal exactly **100.0 for every legitimate URL** and vary for phishing URLs. This single feature perfectly separates classes — not through genuine learning but through dataset construction. The feature was dropped and the experiment moved to a more honest dataset.

### Finding 2 — HTML Feature Bias (PhiUSIIL Dataset)
Phishing pages in PhiUSIIL have a median of **0** across nearly all HTML features (`NoOfImage`, `NoOfJS`, `NoOfCSS`, `NoOfExternalRef`). This reflects a dataset construction bias — phishing pages were scraped as skeleton HTML while legitimate pages were fully loaded. Results were artificially inflated by this structural difference rather than genuine URL-based signals.

### Finding 3 — Domain Shift Between Datasets
A model trained on PhiUSIIL and tested on ISCX-URL-2016 achieved only **23% phishing recall** despite 99%+ accuracy on its own test set. Analysis revealed:

```
IsHTTPS:          PhiUSIIL phishing=0.49  ISCX phishing=0.07
LetterRatio:      PhiUSIIL phishing=0.57  ISCX phishing=0.81
CharContinuation: PhiUSIIL phishing=0.73  ISCX phishing=0.96
```

ISCX phishing URLs structurally resemble legitimate URLs — high letter ratio, clean character distribution, low special chars. The model learned patterns from one distribution that didn't transfer to another. This motivated switching to PhreshPhish, a dataset specifically designed to minimize this bias.

---

## 📦 Dataset

**[PhreshPhish](https://huggingface.co/datasets/phreshphish/phreshphish)** — July 2024 to March 2025

| Split | Total | Phishing | Benign |
|---|---|---|---|
| Train | 498,255 | 221,526 | 276,729 |
| Test | 168,060 | 76,800 | 91,260 |

- Phishing URLs sourced from **PhishTank**, **APWG eCrime eXchange**, and **NetCraft**
- Benign URLs from anonymized browsing telemetry of **6 million real Webroot users**
- Peer-reviewed paper published July 2025
- Specifically designed to minimize leakage and dataset construction bias

---

## ⚙️ Pipeline Architecture

```
Raw URL string
    ↓
URLFeatureExtractor   (extracts 21 features from URL string)
    ↓
StandardScaler        (for Logistic Regression only)
    ↓
XGBoost Classifier    (tuned with Optuna — 30 trials)
    ↓
Phishing / Benign prediction
```

The entire pipeline is a single sklearn `Pipeline` object. Input is a raw URL string. Output is a prediction. No manual feature extraction required at inference time.

```python
pipeline.predict(['https://suspicious-login.xyz/paypal'])
# → array([0])  # 0 = phishing
```

---

## 🛠️ Feature Engineering

All 21 features are extracted purely from the URL string — no HTTP requests, no HTML parsing, no DNS lookups.

| Feature | Description |
|---|---|
| `URLLength` | Total character count |
| `DomainLength` | Domain portion length |
| `TLDLength` | Top-level domain length |
| `IsDomainIP` | Raw IP used instead of domain name |
| `NoOfSubDomain` | Number of subdomains |
| `IsHTTPS` | HTTPS protocol present |
| `HasObfuscation` | Percent-encoded characters present |
| `NoOfObfuscatedChar` | Count of %XX sequences |
| `ObfuscationRatio` | Obfuscated chars / URL length |
| `NoOfLettersInURL` | Alphabetic character count |
| `LetterRatioInURL` | Letters / URL length |
| `NoOfDegitsInURL` | Digit character count |
| `DegitRatioInURL` | Digits / URL length |
| `NoOfEqualsInURL` | Count of `=` characters |
| `NoOfQMarkInURL` | Count of `?` characters |
| `NoOfAmpersandInURL` | Count of `&` characters |
| `NoOfOtherSpecialCharsInURL` | Non-standard special chars |
| `SpacialCharRatioInURL` | Special chars / URL length |
| `URLCharProb` | Character distribution entropy score |
| `CharContinuationRate` | Sequence continuity score |
| `TLDLegitimateProb` | TLD frequency in benign training URLs |

---

## 🔧 Hyperparameter Tuning

Optuna TPE sampler with 30 trials, 3-fold stratified CV on 50K URL subsample.

**Best XGBoost parameters found:**

```python
{
    'n_estimators':     131,
    'max_depth':        9,
    'learning_rate':    0.1247,
    'subsample':        0.9422,
    'colsample_bytree': 0.8141,
    'min_child_weight': 1,
    'gamma':            0.1635,
    'reg_alpha':        4.45e-08,
    'reg_lambda':       0.0106
}
```

Tuning improved AUC from **0.9690 → 0.9697** (+0.0007). The modest gain confirms the default parameters were already near-optimal for this feature set.

---

## 📊 Overfitting Analysis

| Model | Train AUC | Test AUC | Gap | Verdict |
|---|---|---|---|---|
| Logistic Regression | 0.9289 | 0.9244 | 0.0045 | ✅ Good fit |
| Decision Tree | 0.9996 | 0.8671 | 0.1325 | ⚠️ Overfit |
| Random Forest | 0.9993 | 0.9646 | 0.0347 | ✅ Good fit |
| XGBoost | 0.9897 | 0.9690 | 0.0207 | ✅ Good fit |
| LightGBM | 0.9863 | 0.9669 | 0.0194 | ✅ Good fit |

Learning curves show XGBoost and LightGBM are still **converging** — more training data would push validation AUC higher. Decision Tree is excluded from production consideration due to severe overfitting.

---

## 🌍 Real-World Validation

Model tested on live URLs fetched in real time from OpenPhish (active phishing feed) and Tranco/Majestic top sites (legitimate):

```
URLs tested     : 600 (300 phishing + 300 legitimate)
Collected       : March 2026
Inference time  : 0.4ms per URL
```

Key observation: The model struggles with **combosquatting attacks** — URLs that embed real brand names in otherwise clean URL structures (e.g. `roblox.com.ge`, `cliffedekllc.com/bankofamerica/`). These require semantic brand-name analysis beyond pure URL structure features.

---

## 🚀 Quickstart

```bash
git clone https://github.com/yourusername/phishing-url-detection
cd phishing-url-detection
pip install -r requirements.txt
```

```python
import pickle

# Load pipeline
pipeline = pickle.load(open('models/best_pipeline_tuned.pkl', 'rb'))

# Predict any URL
urls = [
    'https://google.com',
    'http://paypal-secure-login.xyz/verify',
    'https://roblox.com.ge/users/12345/profile'
]

preds = pipeline.predict(urls)
proba = pipeline.predict_proba(urls)[:, 0]  # P(phishing)

for url, pred, prob in zip(urls, preds, proba):
    label = '🚨 PHISHING' if pred == 0 else '✅ BENIGN'
    print(f"{label} ({prob:.2%} phishing confidence) — {url}")
```

---

## 📁 Project Structure

```
phishing-url-detection/
├── models/
│   ├── best_pipeline_tuned.pkl     # Best XGBoost pipeline (raw URL → prediction)
│   ├── all_tuned_pipelines.pkl     # All 3 tuned tree model pipelines
│   └── optuna_studies.pkl          # Optuna study objects for analysis
├── reports/
│   ├── phreshphish_results.csv     # Baseline model comparison
│   ├── tuned_results.csv           # Optuna tuned results
│   └── figures/
│       ├── learning_curves.png     # Overfitting analysis plots
│       └── live_test_analysis.png  # Real-world validation plots
├── notebooks/
│   └── phishing_detection.ipynb   # Full Colab notebook
├── requirements.txt
└── README.md
```

---

## 📋 Requirements

```
scikit-learn
xgboost
lightgbm
optuna
datasets
pandas
numpy
matplotlib
seaborn
```

---

## 🔮 Future Work

- **Semantic features** — brand name similarity detection for combosquatting
- **Temporal analysis** — track how phishing URL patterns evolve over time
- **HTML hybrid model** — combine URL features with lightweight HTML signals
- **Early stopping** — add XGBoost early stopping with validation set for faster tuning
- **Streamlit app** — live URL checker with SHAP explanation per prediction
- **API deployment** — FastAPI wrapper for production integration

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- **PhreshPhish dataset** — Webroot Security, 2025
- **PhiUSIIL dataset** — Arvind Prasad & Shalini Chandra, UCI ML Repository, 2024
- **ISCX-URL-2016** — Canadian Institute for Cybersecurity, University of New Brunswick
