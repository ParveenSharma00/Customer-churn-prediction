<p align="center">
  <img src="https://img.shields.io/badge/Kaggle-Rank%20%232-gold?style=for-the-badge&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/AUC--ROC-0.91405-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

# 🏆 Customer Churn Prediction — Kaggle Playground Series S6E3

> **End-to-end Machine Learning pipeline** predicting telecom customer churn.  
> Achieved **AUC-ROC: 0.91405** and **Rank #2** on the public leaderboard.

---

## 📌 Competition Overview

| Detail | Info |
|--------|------|
| **Competition** | [Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) |
| **Task** | Predict the probability of customer churn (binary classification) |
| **Metric** | AUC-ROC (Area Under ROC Curve) |
| **Training Data** | 594,194 customers × 19 features |
| **Test Data** | 254,655 customers |
| **Result** | **0.91405 AUC — Rank #2** |

---

## 🗂️ Repository Structure

```
kaggle-churn-prediction/
│
├── README.md                              ← You are here
├── requirements.txt                       ← Python dependencies
├── .gitignore                             ← Files to exclude from git
│
├── data/
│   ├── README.md                          ← Download instructions
│   └── sample_submission.csv              ← Kaggle submission format
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb     ← Phase 1-5: EDA + Feature Engineering
│   ├── 02_Baseline_Models.ipynb           ← Part 1: Logistic Regression + Random Forest
│   ├── 03_Boosting_Models.ipynb           ← Part 2: XGBoost + LightGBM + CatBoost
│   ├── 04_Hyperparameter_Tuning.ipynb     ← Part 3: Optuna Bayesian Optimization
│   ├── 05_Advanced_Techniques.ipynb       ← Part 4: Multi-seed + Feature Selection
│   └── 06_Ensemble_and_Submission.ipynb   ← Part 5: 6 Ensemble Methods + Submission
│
├── src/
│   ├── preprocessing_pipeline.py          ← Standalone preprocessing script
│   └── final_summary.py                   ← Final results summary
│
└── submissions/
    └── submission_rank1.csv               ← Best submission (0.91405)
```

---

## 🔬 Methodology

### The Complete Pipeline

```
Raw Data → EDA → Preprocessing → Feature Engineering → Modeling → Tuning → Ensemble → Submit
```

### Phase 1-2: Problem Understanding & Data Loading

- Telecom dataset with **594K customers** — demographics, services, billing, contract info
- Target: `Churn` (Yes/No) — **22.5% churn rate** (moderately imbalanced, 3.44:1 ratio)
- Zero missing values, zero duplicates — clean synthetic dataset

### Phase 3: Exploratory Data Analysis

Performed **univariate**, **bivariate**, and **multivariate** analysis to discover patterns:

| Analysis | Key Discovery |
|----------|--------------|
| **Univariate** | `tenure` is bimodal (new vs loyal customers), `TotalCharges` right-skewed |
| **Bivariate** | Month-to-month contracts → **42% churn** vs 3% for two-year contracts |
| **Multivariate** | Fiber optic + no protection services + new customer = **55%+ churn rate** |

**Top churn predictors identified:**
1. 📄 **Contract type** — Month-to-month = 94% of all churners
2. 🌐 **Internet service** — Fiber optic = 85% of churners  
3. ⏱️ **Tenure** — Churners average 17 months vs 42 for loyal customers
4. 💳 **Payment method** — Electronic check = 79% of churners
5. 🛡️ **Online security** — No security = 42% churn rate

### Phase 4-5: Preprocessing & Feature Engineering

Engineered **29 new features** across 7 categories, expanding from 19 → 71 total features:

| Category | Count | Features | Insight |
|----------|-------|----------|---------|
| **Service-based** | 6 | `n_addons`, `n_protection`, `has_streaming` | More services = less churn |
| **Account** | 5 | `is_auto_pay`, `is_month_to_month`, `contract_risk` | Commitment level matters |
| **Tenure-based** | 5 | `is_new`, `is_loyal`, `tenure_squared` | Non-linear churn decay |
| **Billing** | 5 | `charge_diff`, `log_total_charges`, `charge_ratio` | Price change detection |
| **Demographics** | 3 | `has_family`, `senior_single`, `family_size` | Family = stability |
| **Interaction** | 4 | `fiber_no_protect`, `tenure_x_charges`, `mtm_and_new` | Combined risk effects |
| **Composite** | 1 | `risk_score` (0-6) | Single most powerful feature |

**Risk Score Impact:**

| Score | Churn Rate | Customer Profile |
|-------|-----------|-----------------|
| 0 (safest) | ~2% | Two-year contract, auto-pay, protection services, loyal |
| 6 (riskiest) | ~73% | Month-to-month, e-check, fiber, no protection, new |

### Phase 6: Model Building (5-Fold Stratified CV)

| Model | AUC-ROC | Type | Key Strength |
|-------|---------|------|-------------|
| Logistic Regression | ~0.912 | Linear baseline | Fast, interpretable |
| Random Forest | ~0.912 | Bagging (parallel trees) | Robust, no scaling needed |
| XGBoost | ~0.914 | Gradient Boosting | Industry standard, regularized |
| LightGBM | ~0.914 | Gradient Boosting | Fastest, leaf-wise growth |
| CatBoost | ~0.916 | Gradient Boosting | Best out-of-box, ordered boosting |

### Phase 7: Tuning & Ensemble

**Hyperparameter tuning** with Optuna (Bayesian optimization, 50 trials per model):
- Tuned XGBoost, LightGBM, and CatBoost independently
- Retrained best parameters with full 5-fold CV

**Advanced techniques:**
- Multi-seed averaging (5 seeds × 5 folds = 25 models)
- Importance-based feature selection

**Ensemble methods applied:**

| Method | Description |
|--------|------------|
| Simple Average | Equal weight to all models |
| Weighted Average | Weight by CV score |
| Rank Average | Average ranks instead of probabilities (most robust) |
| Optimized Weights | scipy.optimize finds best weights |
| Stacking | Meta-learner (Logistic Regression) on OOF predictions |
| Hill Climbing | Greedy weight optimization |

### Final Result: **0.91405 AUC-ROC (Rank #2)**

---

## 📊 Feature Importance

Top 15 most important features (LightGBM):

| Rank | Feature | Type | Importance |
|------|---------|------|-----------|
| 1 | `TotalCharges` | Original | ██████████████████████████ |
| 2 | `MonthlyCharges` | Original | █████████████████████ |
| 3 | `avg_monthly_charge` | ⭐ Engineered | ███████████████████ |
| 4 | `tenure` | Original | ████████████████ |
| 5 | `charge_diff` | ⭐ Engineered | ███████████████ |
| 6 | `charge_ratio` | ⭐ Engineered | ██████████████ |
| 7 | `tenure_x_charges` | ⭐ Engineered | ████████████ |
| 8 | `log_total_charges` | ⭐ Engineered | ██████ |
| 9 | `risk_score` | ⭐ Engineered | ████ |
| 10 | `log_monthly_charges` | ⭐ Engineered | ████ |

**7 out of top 10 features are engineered** — feature engineering was critical to performance.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn |
| **ML Models** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Optimization** | Optuna (Bayesian hyperparameter tuning) |
| **Ensemble** | scipy.optimize, custom stacking |

---

## 🚀 How to Run

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/ParveenSharma00/kaggle-churn-prediction.git
cd kaggle-churn-prediction

# Install dependencies
pip install -r requirements.txt
```

### Get the Data

Download from [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s6e3/data) and place in `data/` folder:
- `train.csv`
- `test.csv`  
- `sample_submission.csv`

### Run the Notebooks (in order)

```
notebooks/01_EDA_and_Preprocessing.ipynb    → Outputs: X_train.csv, X_test.csv, y_train.csv
notebooks/02_Baseline_Models.ipynb          → Outputs: part1_oof_*.npy, part1_test_*.npy
notebooks/03_Boosting_Models.ipynb          → Outputs: part2_oof_*.npy, part2_test_*.npy
notebooks/04_Hyperparameter_Tuning.ipynb    → Outputs: part3_oof_*.npy, part3_test_*.npy
notebooks/05_Advanced_Techniques.ipynb      → Outputs: part4_oof_*.npy, part4_test_*.npy
notebooks/06_Ensemble_and_Submission.ipynb  → Outputs: submission_final.csv
```

### Or Run Standalone Scripts

```bash
# Run preprocessing pipeline
python src/preprocessing_pipeline.py

# View final results summary
python src/final_summary.py
```

---

## 📚 What I Learned

This project was a deep-dive into the complete ML workflow. Key concepts mastered:

| Area | Concepts |
|------|----------|
| **EDA** | Univariate/Bivariate/Multivariate analysis, distribution shapes, correlation |
| **Feature Engineering** | Domain features, interactions, composite scores, log transforms |
| **Preprocessing** | Label/Ordinal/One-Hot encoding, scaling, handling imbalance |
| **Cross-Validation** | Stratified K-Fold, OOF predictions, data leakage prevention |
| **Models** | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| **Tuning** | Optuna Bayesian optimization, early stopping, regularization |
| **Ensemble** | Averaging, weighted avg, rank avg, stacking, hill climbing |
| **Evaluation** | AUC-ROC, accuracy paradox, class imbalance handling |

---

## 🔑 Key Takeaways

1. **Feature engineering > model selection** — Engineered features dominated the importance rankings
2. **Ensemble always helps** — Combining 3+ diverse models consistently beats any single model
3. **Domain knowledge matters** — Understanding telecom churn patterns guided better feature creation
4. **CV score ≈ LB score** — Proper stratified CV gave reliable estimates (0.914 CV → 0.914 LB)
5. **Simple baselines are powerful** — Logistic Regression at 0.912 was surprisingly close to boosting at 0.916

---

## 📬 Author

**Parveen Sharma**  
Senior Data Analyst | ML Practitioner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/parveen-150)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/ParveenSharma00)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/)

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for hosting the Playground Series competitions
- Original dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Built with guidance from Claude (Anthropic) as a learning exercise

---

<p align="center">
  <b>⭐ If you found this useful, please give it a star!</b>
</p>
