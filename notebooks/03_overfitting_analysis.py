import os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve, train_test_split

# 1. Load Data & Pipelines

print("Loading PhreshPhish...")
df_train = pd.read_csv('data/df_train.csv.gz', compression="gzip")
df_test  = pd.read_csv('data/df_test.csv')

df_train['binary_label'] = (df_train['label'] == 'benign').astype(int)
df_test['binary_label'] = (df_test['label'] == 'benign').astype(int)

train_urls = df_train['url'].values
y_train    = df_train['binary_label'].values
test_urls  = df_test['url'].values
y_test     = df_test['binary_label'].values

# Load baseline pipelines (already fitted in notebook 01)
PIPELINES = pickle.load(open('models/baseline_pipelines.pkl', 'rb'))
print(f"Loaded {len(PIPELINES)} baseline pipelines")

# Part-1 : Train vs Test AUC

print("=" * 68)
print("PART-1 :” Train vs Test Scores")
print("=" * 68)
print(f"\n{'Model':<22} {'Train AUC':>10} {'Test AUC':>10} "
      f"{'Gap':>8} {'Verdict':>15}")
print("─"* 68)

fit_results = {}

for name, pipeline in PIPELINES.items():
    train_proba = pipeline.predict_proba(train_urls)[:, 0]
    test_proba  = pipeline.predict_proba(test_urls)[:, 0]

    train_auc = roc_auc_score(1 - y_train, train_proba)
    test_auc  = roc_auc_score(1 - y_test,  test_proba)
    gap       = train_auc - test_auc

    if gap > 0.05:
        verdict = 'Overfit'
    elif test_auc < 0.80:
        verdict = 'Underfit'
    else:
        verdict = 'Good fit'

    fit_results[name] = {
        'Train AUC': round(train_auc, 6),
        'Test AUC':  round(test_auc,  6),
        'Gap':       round(gap, 6),
        'Verdict':   verdict
    }
    print(f"{name:<22} {train_auc:>10.4f} {test_auc:>10.4f} "
          f"{gap:>8.4f} {verdict:>15}")

print(f"\n{'─'*68}")
print("\nGap interpretation:")
print("  < 0.01     - Almost no overfitting")
print("  0.01-0.05  - Slight overfitting (acceptable)")
print("  > 0.05     - Significant overfitting")
print("  AUC < 0.80 - Underfitting")

# Part 2 : Learning Curves

print("\n" + "=" * 68)
print("PART 2 : Learning Curves")
print("Computing... (takes a few minutes)")
print("=" * 68)

urls_lc, _, y_lc, _ = train_test_split(
    train_urls, y_train,
    train_size=30_000,
    stratify=y_train,
    random_state=42
)

LC_MODELS = {k: v for k, v in PIPELINES.items()
             if k in ['Logistic Regression', 'XGBoost', 'LightGBM']}

train_sizes = np.linspace(0.1, 1.0, 8)
lc_results  = {}

for name, pipeline in LC_MODELS.items():
    print(f"  Computing {name}...")
    sizes, train_scores, val_scores = learning_curve(
        pipeline, urls_lc, y_lc,
        train_sizes=train_sizes,
        cv=3,
        scoring='roc_auc',
        n_jobs=1,
        shuffle=True,
        random_state=42
    )
    lc_results[name] = {
        'sizes':      sizes,
        'train_mean': train_scores.mean(axis=1),
        'train_std':  train_scores.std(axis=1),
        'val_mean':   val_scores.mean(axis=1),
        'val_std':    val_scores.std(axis=1),
    }
    print(f"    Done âœ…")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {
    'Logistic Regression': '#3498db',
    'XGBoost':             '#e74c3c',
    'LightGBM':            '#2ecc71'
}

for ax, (name, lc) in zip(axes, lc_results.items()):
    color = colors[name]
    ax.plot(lc['sizes'], lc['train_mean'],
            color=color, label='Train AUC', linewidth=2, marker='o')
    ax.fill_between(lc['sizes'],
                    lc['train_mean'] - lc['train_std'],
                    lc['train_mean'] + lc['train_std'],
                    alpha=0.15, color=color)
    ax.plot(lc['sizes'], lc['val_mean'],
            color=color, label='Val AUC', linewidth=2,
            marker='s', linestyle='--')
    ax.fill_between(lc['sizes'],
                    lc['val_mean'] - lc['val_std'],
                    lc['val_mean'] + lc['val_std'],
                    alpha=0.15, color=color)

    final_gap = lc['train_mean'][-1] - lc['val_mean'][-1]
    ax.set_title(f"{name}\nFinal Gap: {final_gap:.4f}", fontweight='bold')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0.90, color='gray', linestyle=':', alpha=0.5)

plt.suptitle('Learning Curves - Overfitting / Underfitting Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Part 3 : Bias-Variance Summary

print("\n" + "=" * 68)
print("PART 3 : Bias-Variance Summary")
print("=" * 68)

for name, lc in lc_results.items():
    train_final = lc['train_mean'][-1]
    val_final   = lc['val_mean'][-1]
    gap         = train_final - val_final
    converging  = (lc['val_mean'][-1] - lc['val_mean'][-3]) > 0.001

    print(f"\n      {name}      )
    print(f"  Train AUC (full data) : {train_final:.4f}")
    print(f"  Val   AUC (full data) : {val_final:.4f}")
    print(f"  Gap                   : {gap:.4f}")

    if gap > 0.05:
        print("   HIGH VARIANCE (overfitting)")
        print("    Fix: increase regularization, reduce max_depth, add more data")
    elif val_final < 0.85:
        print("   HIGH BIAS (underfitting)")
        print("    Fix: increase n_estimators, increase max_depth, add features")
    elif converging:
        print("   CONVERGING: more data would still help")
    else:
        print("   GOOD FIT: stable train/val gap, model well calibrated")
