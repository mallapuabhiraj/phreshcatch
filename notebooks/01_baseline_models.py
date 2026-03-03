import os, time, pickle
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sys
sys.path.append('..')
from src.features import URLFeatureExtractor

os.makedirs('models',          exist_ok=True)
os.makedirs('reports',         exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# 1. Load PhreshPhish

print("Loading PhreshPhish v1.0.1...")

df_train = pd.read_csv('data/df_train.csv.gz', compression="gzip")
df_test  = pd.read_csv('data/df_test.csv')

# Label convention: 0 = phishing, 1 = benign
df_train['binary_label'] = (df_train['label'] == 'benign').astype(int)
df_test['binary_label']  = (df_test['label']  == 'benign').astype(int)

y_train = df_train['binary_label'].values
y_test  = df_test['binary_label'].values

print(f"        Train Split        ")
print(f"  Total    : {df_train.shape[0]:,}")
print(f"  Benign   : {(df_train['binary_label']==1).sum():,}")
print(f"  Phishing : {(df_train['binary_label']==0).sum():,}")

print(f"         Test Split       ")
print(f"  Total    : {df_test.shape[0]:,}")
print(f"  Benign   : {(df_test['binary_label']==1).sum():,}")
print(f"  Phishing : {(df_test['binary_label']==0).sum():,}")

# 2. Build Whitelist (Tranco Top 10K)

import requests, zipfile, io

TRUSTED_DOMAINS = set()

print("Loading Tranco Top 10K...")
try:
    r = requests.get(
        'https://tranco-list.eu/top-1m.csv.zip',
        timeout=30,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                parts = line.decode().strip().split(',')
                if len(parts) >= 2 and '.' in parts[1]:
                    TRUSTED_DOMAINS.add(parts[1].strip().lower())
    print(f"Tranco loaded: {len(TRUSTED_DOMAINS):,} domains")
except Exception as e:
    print(f"Tranco failed: {e}")

pickle.dump(TRUSTED_DOMAINS, open('models/trusted_domains.pkl', 'wb'))
print("Saved at models/trusted_domains.pkl")

# 3. Define Pipelines

train_urls = df_train['url'].values
test_urls  = df_test['url'].values

PIPELINES = {
    'Logistic Regression': Pipeline([
        ('extractor', URLFeatureExtractor(trusted_domains=TRUSTED_DOMAINS)),
        ('scaler',    StandardScaler()),
        ('model',     LogisticRegression(max_iter=300, random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('extractor', URLFeatureExtractor(trusted_domains=TRUSTED_DOMAINS)),
        ('model',     DecisionTreeClassifier(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('extractor', URLFeatureExtractor(trusted_domains=TRUSTED_DOMAINS)),
        ('model',     RandomForestClassifier(n_estimators=100,
                                              n_jobs=-1, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('extractor', URLFeatureExtractor(trusted_domains=TRUSTED_DOMAINS)),
        ('model',     XGBClassifier(n_estimators=200, eval_metric='logloss',
                                     n_jobs=-1, random_state=42, verbosity=0))
    ]),
    'LightGBM': Pipeline([
        ('extractor', URLFeatureExtractor(trusted_domains=TRUSTED_DOMAINS)),
        ('model',     LGBMClassifier(n_estimators=200, n_jobs=-1,
                                      random_state=42, verbose=-1))
    ]),
}

# 4. Train & Evaluate

results = {}

print("=== Training All Pipelines on PhreshPhish ===\n")
print(f"{'Model':<22} {'Acc %':>8} {'F1 %':>8} {'AUC':>8} "
      f"{'PRAUC':>8} {'FP':>7} {'FN':>7} {'Time':>7}")
print("─" * 55)

for name, pipeline in PIPELINES.items():
    t0 = time.time()
    pipeline.fit(train_urls, y_train)
    elapsed = time.time() - t0

    preds = pipeline.predict(test_urls)
    proba = pipeline.predict_proba(test_urls)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    acc   = accuracy_score(y_test, preds) * 100
    f1    = f1_score(y_test, preds) * 100
    auc   = roc_auc_score(y_test, proba)
    prauc = average_precision_score(y_test, proba)

    results[name] = {
        'Accuracy (%)': round(acc, 4),
        'F1 (%)':       round(f1, 4),
        'ROC-AUC':      round(auc, 6),
        'PR-AUC':       round(prauc, 6),
        'FP':           int(fp),
        'FN':           int(fn),
        'Time':         f"{elapsed:.1f}s"
    }
    print(f"{name:<22} {acc:>8.2f} {f1:>8.2f} {auc:>8.4f} {prauc:>8.4f} "
          f"{fp:>7,} {fn:>7,} {elapsed:>6.1f}s")

results_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
print(f"\n{'─'*75}")
print("\n=== FINAL RANKING ===")
print(results_df.to_string())
