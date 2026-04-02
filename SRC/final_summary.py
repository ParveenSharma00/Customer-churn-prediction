"""
╔══════════════════════════════════════════════════════════════════════╗
║       KAGGLE PLAYGROUND S6E3 — COMPLETE PIPELINE SUMMARY            ║
║       Customer Churn Prediction — Final Report                      ║
╚══════════════════════════════════════════════════════════════════════╝

Run this AFTER all Parts 1-5 notebooks to see the complete picture.
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

# ============================================================
# LOAD TARGET
# ============================================================
y_train = pd.read_csv('y_train.csv').squeeze()
test_ids = pd.read_csv('test_ids.csv').squeeze()

print("╔" + "═" * 68 + "╗")
print("║" + "  🏆 COMPLETE MODEL PIPELINE — FINAL SUMMARY".center(68) + "║")
print("║" + "  Kaggle Playground S6E3 | Customer Churn Prediction".center(68) + "║")
print("╚" + "═" * 68 + "╝")

# ============================================================
# LOAD ALL OOF PREDICTIONS & CALCULATE SCORES
# ============================================================
print("\n" + "█" * 70)
print("█  SECTION 1: ALL INDIVIDUAL MODELS")
print("█" * 70)

all_models = {}

model_files = [
    # (Display Name, Category, OOF file, Test file)
    ("Logistic Regression",    "Part 1: Baseline",    "part1_oof_lr.npy",           "part1_test_lr.npy"),
    ("Random Forest",          "Part 1: Baseline",    "part1_oof_rf.npy",           "part1_test_rf.npy"),
    ("XGBoost (default)",      "Part 2: Boosting",    "part2_oof_xgb.npy",          "part2_test_xgb.npy"),
    ("LightGBM (default)",     "Part 2: Boosting",    "part2_oof_lgb.npy",          "part2_test_lgb.npy"),
    ("CatBoost (default)",     "Part 2: Boosting",    "part2_oof_cb.npy",           "part2_test_cb.npy"),
    ("XGBoost (Optuna-tuned)", "Part 3: Tuned",       "part3_oof_xgb.npy",          "part3_test_xgb.npy"),
    ("LightGBM (Optuna-tuned)","Part 3: Tuned",       "part3_oof_lgb.npy",          "part3_test_lgb.npy"),
    ("CatBoost (Optuna-tuned)","Part 3: Tuned",       "part3_oof_cb.npy",           "part3_test_cb.npy"),
    ("LightGBM (Multi-seed)",  "Part 4: Advanced",    "part4_oof_multiseed_lgb.npy","part4_test_multiseed_lgb.npy"),
    ("LightGBM (Feat-select)", "Part 4: Advanced",    "part4_oof_selected_lgb.npy", "part4_test_selected_lgb.npy"),
]

print(f"\n  {'#':<4} {'Model':<30} {'Part':<20} {'AUC-ROC':>10} {'Status'}")
print(f"  {'─' * 75}")

loaded_count = 0
for i, (name, part, oof_file, test_file) in enumerate(model_files, 1):
    if os.path.exists(oof_file) and os.path.exists(test_file):
        oof = np.load(oof_file)
        test = np.load(test_file)
        score = roc_auc_score(y_train, oof)
        all_models[name] = {'oof': oof, 'test': test, 'score': score, 'part': part}
        loaded_count += 1
        print(f"  {i:<4} {name:<30} {part:<20} {score:.6f}   ✅ Loaded")
    else:
        print(f"  {i:<4} {name:<30} {part:<20} {'---':>10}   ⚠️ Not found")

print(f"\n  Models loaded: {loaded_count}/{len(model_files)}")

# ============================================================
# SINGLE MODEL LEADERBOARD
# ============================================================
if all_models:
    print("\n" + "█" * 70)
    print("█  SECTION 2: SINGLE MODEL LEADERBOARD")
    print("█" * 70)
    
    sorted_models = sorted(all_models.items(), key=lambda x: x[1]['score'], reverse=True)
    medals = ['🥇', '🥈', '🥉']
    
    print(f"\n  {'Rank':<6} {'Model':<35} {'AUC-ROC':>10} {'Part'}")
    print(f"  {'━' * 65}")
    
    for rank, (name, data) in enumerate(sorted_models):
        medal = medals[rank] if rank < 3 else '  '
        flag = " ← BEST SINGLE" if rank == 0 else ""
        print(f"  {medal} {rank+1:<3} {name:<33} {data['score']:.6f}   {data['part']}{flag}")
    
    best_single = sorted_models[0]
    worst_single = sorted_models[-1]
    improvement = best_single[1]['score'] - worst_single[1]['score']
    
    print(f"\n  Best:  {best_single[0]} ({best_single[1]['score']:.6f})")
    print(f"  Worst: {worst_single[0]} ({worst_single[1]['score']:.6f})")
    print(f"  Gap:   {improvement:.6f} ({improvement*100:.4f}%)")

# ============================================================
# ENSEMBLES
# ============================================================
if len(all_models) >= 2:
    print("\n" + "█" * 70)
    print("█  SECTION 3: ENSEMBLE METHODS")
    print("█" * 70)
    
    ensemble_results = {}
    
    # ── 3A: Simple Average ──
    oof_all = np.mean([d['oof'] for d in all_models.values()], axis=0)
    test_all = np.mean([d['test'] for d in all_models.values()], axis=0)
    score_all = roc_auc_score(y_train, oof_all)
    ensemble_results['Simple Average (all)'] = {'score': score_all, 'test': test_all}
    
    # ── 3B: Boost-only models ──
    boost_names = [n for n in all_models if any(x in n for x in ['XGBoost', 'LightGBM', 'CatBoost'])]
    if len(boost_names) >= 2:
        oof_boost = np.mean([all_models[n]['oof'] for n in boost_names], axis=0)
        test_boost = np.mean([all_models[n]['test'] for n in boost_names], axis=0)
        score_boost = roc_auc_score(y_train, oof_boost)
        ensemble_results[f'Boost Average ({len(boost_names)} models)'] = {'score': score_boost, 'test': test_boost}
    
    # ── 3C: Weighted Average ──
    total_w = sum(d['score'] for d in all_models.values())
    oof_w = sum(d['oof'] * d['score'] / total_w for d in all_models.values())
    test_w = sum(d['test'] * d['score'] / total_w for d in all_models.values())
    score_w = roc_auc_score(y_train, oof_w)
    ensemble_results['Weighted Average'] = {'score': score_w, 'test': test_w}
    
    # ── 3D: Rank Average ──
    from scipy.stats import rankdata
    oof_ranks = np.mean([rankdata(d['oof']) for d in all_models.values()], axis=0)
    test_ranks = np.mean([rankdata(d['test']) for d in all_models.values()], axis=0)
    oof_rank_norm = (oof_ranks - oof_ranks.min()) / (oof_ranks.max() - oof_ranks.min())
    test_rank_norm = (test_ranks - test_ranks.min()) / (test_ranks.max() - test_ranks.min())
    score_rank = roc_auc_score(y_train, oof_rank_norm)
    ensemble_results['Rank Average'] = {'score': score_rank, 'test': test_rank_norm}
    
    # ── 3E: Top-3 Only ──
    top3_names = [n for n, _ in sorted_models[:3]]
    oof_top3 = np.mean([all_models[n]['oof'] for n in top3_names], axis=0)
    test_top3 = np.mean([all_models[n]['test'] for n in top3_names], axis=0)
    score_top3 = roc_auc_score(y_train, oof_top3)
    ensemble_results['Top-3 Average'] = {'score': score_top3, 'test': test_top3}
    
    # ── 3F: Tuned-only (if available) ──
    tuned_names = [n for n in all_models if 'tuned' in n.lower() or 'Optuna' in n]
    if len(tuned_names) >= 2:
        oof_tuned = np.mean([all_models[n]['oof'] for n in tuned_names], axis=0)
        test_tuned = np.mean([all_models[n]['test'] for n in tuned_names], axis=0)
        score_tuned = roc_auc_score(y_train, oof_tuned)
        ensemble_results[f'Tuned-only Average ({len(tuned_names)})'] = {'score': score_tuned, 'test': test_tuned}
    
    sorted_ens = sorted(ensemble_results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print(f"\n  {'Rank':<6} {'Ensemble Method':<40} {'AUC-ROC':>10}")
    print(f"  {'━' * 58}")
    for rank, (name, data) in enumerate(sorted_ens):
        medal = medals[rank] if rank < 3 else '  '
        flag = " ← BEST ENSEMBLE" if rank == 0 else ""
        print(f"  {medal} {rank+1:<3} {name:<38} {data['score']:.6f}{flag}")

# ============================================================
# GRAND FINAL LEADERBOARD
# ============================================================
print("\n" + "█" * 70)
print("█  SECTION 4: 🏆 GRAND FINAL LEADERBOARD")
print("█" * 70)

grand = {}
for name, data in all_models.items():
    grand[f"[Single] {name}"] = (data['score'], data['test'])
if 'ensemble_results' in dir() or 'ensemble_results' in locals():
    for name, data in ensemble_results.items():
        grand[f"[Ensemble] {name}"] = (data['score'], data['test'])

sorted_grand = sorted(grand.items(), key=lambda x: x[1][0], reverse=True)

print(f"\n  {'Rank':<6} {'Model / Ensemble':<50} {'AUC-ROC':>10}")
print(f"  {'━' * 68}")
for rank, (name, (score, _)) in enumerate(sorted_grand):
    medal = medals[rank] if rank < 3 else '  '
    flag = ""
    if rank == 0:
        flag = " ★ SUBMIT THIS"
    elif rank == 1:
        flag = " ★ 2nd choice"
    elif rank == 2:
        flag = " ★ 3rd choice"
    print(f"  {medal} {rank+1:>2}. {name:<48} {score:.6f}{flag}")

# ============================================================
# SUBMISSION FILES
# ============================================================
print("\n" + "█" * 70)
print("█  SECTION 5: SUBMISSION FILES")
print("█" * 70)

# Generate submissions from top 3
for rank in range(min(3, len(sorted_grand))):
    name = sorted_grand[rank][0]
    score = sorted_grand[rank][1][0]
    test_preds = sorted_grand[rank][1][1]
    
    filename = f"submission_rank{rank+1}.csv"
    sub = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
    sub.to_csv(filename, index=False)
    
    tag = "⭐ PRIMARY" if rank == 0 else "⭐ BACKUP" if rank == 1 else "  SAFE"
    print(f"\n  {tag}  {filename}")
    print(f"          Model: {name}")
    print(f"          CV AUC: {score:.6f}")
    print(f"          Pred range: {test_preds.min():.4f} → {test_preds.max():.4f}")
    print(f"          Pred mean:  {test_preds.mean():.4f}")

# ============================================================
# PIPELINE OVERVIEW
# ============================================================
print("\n" + "█" * 70)
print("█  SECTION 6: COMPLETE PIPELINE OVERVIEW")
print("█" * 70)

print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    WHAT WE BUILT                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Phase 1  │ Problem Understanding                               │
  │  Phase 2  │ Data Loading & First Look                           │
  │  Phase 3  │ EDA: Univariate → Bivariate → Multivariate         │
  │  Phase 4  │ Data Preprocessing & Cleaning                       │
  │  Phase 5  │ Feature Engineering (29 new features, 71 total)     │
  │           │                                                      │
  │  Part 1   │ Baseline: Logistic Regression + Random Forest       │
  │  Part 2   │ Gradient Boosting: XGBoost + LightGBM + CatBoost   │
  │  Part 3   │ Optuna Tuning: 50 trials per model                 │
  │  Part 4   │ Advanced: Multi-seed + Feature Selection            │
  │  Part 5   │ Ensemble: 6 methods → Final Submission             │
  │                                                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │                    BY THE NUMBERS                                │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Training rows:        594,194                                  │
  │  Test rows:            254,655                                  │
  │  Original features:    19                                       │
  │  Engineered features:  29                                       │
  │  Total features:       71                                       │
  │  Models trained:       10+                                      │
  │  Ensemble methods:     6                                        │
  │  Submission files:     3                                        │
  │                                                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │                    CONCEPTS LEARNED                              │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  EDA          │ Histogram, KDE, Box Plot, ECDF, Heatmap        │
  │  Statistics   │ Skewness, Kurtosis, IQR, Correlation           │
  │  Preprocessing│ Encoding (Label, Ordinal, One-Hot), Scaling    │
  │  FE          │ Interactions, Binning, Log Transform, Composite │
  │  CV          │ Stratified K-Fold, OOF predictions              │
  │  Models      │ LogReg, RF, XGBoost, LightGBM, CatBoost        │
  │  Tuning      │ Optuna Bayesian Optimization                    │
  │  Ensemble    │ Averaging, Weighted, Rank, Stacking, Hill Climb │
  │  Metric      │ AUC-ROC, why not accuracy                       │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# FINAL MESSAGE
# ============================================================
if sorted_grand:
    best = sorted_grand[0]
    print(f"  🏆 BEST SCORE: {best[1][0]:.6f}")
    print(f"     Model: {best[0]}")
    print(f"     File:  submission_rank1.csv")

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   📤 SUBMISSION STRATEGY:                                       │
  │                                                                  │
  │   1. Upload submission_rank1.csv  (best CV score)               │
  │   2. Upload submission_rank2.csv  (backup)                      │
  │   3. Upload submission_rank3.csv  (safe/robust pick)            │
  │                                                                  │
  │   For FINAL 2 selections on Kaggle:                             │
  │   → Select #1: Best public LB score                             │
  │   → Select #2: submission with Rank Average (most stable)       │
  │                                                                  │
  │   Good luck on the leaderboard! 🚀                              │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
""")
