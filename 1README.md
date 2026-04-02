<p align="center">
  <img src="https://img.shields.io/badge/Kaggle-Top%2028%25-blue?style=for-the-badge&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/AUC--ROC-0.91405-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Rank-1400%20%2F%205000+-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-10+-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Features-71-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<h1 align="center">Customer Churn Prediction</h1>
<h3 align="center">Kaggle Playground Series S6E3 — Complete End-to-End ML Pipeline</h3>

<p align="center">
  <i>From raw data to Kaggle submission — EDA, Feature Engineering, 5 ML Models, Optuna Tuning, 6 Ensemble Methods</i>
</p>

---

## Competition Overview

A telecom company has **594,194 customers**. Some customers **leave** (churn) and go to competitors. The goal is to **predict the probability of churn** for 254,655 test customers.

| Detail | Info |
|--------|------|
| **Competition** | [Playground Series S6E3 — Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3) |
| **Task** | Binary Classification — predict churn probability (0 to 1) |
| **Metric** | AUC-ROC (Area Under Receiver Operating Characteristic Curve) |
| **Training Data** | 594,194 customers x 19 features |
| **Test Data** | 254,655 customers |
| **My Score** | **0.91405 AUC-ROC** |
| **My Rank** | **~1400 / 5000+ participants (Top 28%)** |

---

## Repository Structure

```
kaggle-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── README.md                          # Data download instructions
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb     # Phase 1-5: Full EDA + Feature Engineering
│   ├── 02_Baseline_Models.ipynb           # Logistic Regression + Random Forest
│   ├── 03_Boosting_Models.ipynb           # XGBoost + LightGBM + CatBoost
│   ├── 04_Hyperparameter_Tuning.ipynb     # Optuna Bayesian Optimization
│   ├── 05_Advanced_Techniques.ipynb       # Multi-seed averaging + Feature Selection
│   └── 06_Ensemble_and_Submission.ipynb   # 6 Ensemble methods + Final Submission
├── src/
│   ├── preprocessing_pipeline.py          # Standalone preprocessing script
│   └── final_summary.py                   # Complete results summary
├── charts/                                # All EDA visualizations (13 charts)
└── submissions/
    └── submission_rank1.csv               # Best submission (0.91405)
```

---

## Complete Methodology

### Pipeline Overview

```
Raw Data --> EDA --> Preprocessing --> Feature Engineering --> Modeling --> Tuning --> Ensemble --> Submit
 594K rows   13 charts   Clean data     19 to 71 feats      5 models    Optuna     6 methods   0.914
```

---

### Phase 1-2: Data Understanding

The dataset represents a telecom company's customer database with 4 groups of information:

| Group | Features | Question |
|-------|----------|----------|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents | WHO is the customer? |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | WHAT do they use? |
| **Account** | tenure, Contract, PaperlessBilling, PaymentMethod | HOW are they engaged? |
| **Billing** | MonthlyCharges, TotalCharges | HOW MUCH do they pay? |

**Key Facts:** Zero missing values, zero duplicates, 77.5% stayed vs 22.5% churned (3.44:1 imbalance)

---

### Phase 3: Exploratory Data Analysis

#### Univariate Analysis — One variable at a time

<p align="center">
  <img src="charts/eda_01_univariate_numerical.png" width="100%" />
</p>

**Findings:** tenure is bimodal (new vs loyal), MonthlyCharges bimodal (budget vs premium), TotalCharges right-skewed

<p align="center">
  <img src="charts/eda_02_univariate_categorical.png" width="100%" />
</p>

**Findings:** 50% are month-to-month contracts (highest churn risk), 94% have phone service (low variance)

<p align="center">
  <img src="charts/eda_03_target_distribution.png" width="100%" />
</p>

**Target:** 77.5% Stayed vs 22.5% Left — moderate imbalance, use AUC-ROC not accuracy

#### Bivariate Analysis — Each feature vs Churn

<p align="center">
  <img src="charts/eda_04_bivariate_numerical.png" width="100%" />
</p>

**Key Discovery:** Churners average 17 months tenure vs 42 for loyal customers. Higher monthly charges correlate with more churn.

<p align="center">
  <img src="charts/eda_05_bivariate_categorical.png" width="100%" />
</p>

**Top Churn Predictors:**

| Rank | Feature | High Churn Category | Churn Rate |
|------|---------|-------------------|-----------|
| 1 | **Contract** | Month-to-month | ~42% (vs 3% for Two-year) |
| 2 | **InternetService** | Fiber optic | ~42% |
| 3 | **PaymentMethod** | Electronic check | ~45% |
| 4 | **OnlineSecurity** | No | ~42% |
| 5 | **TechSupport** | No | ~42% |

<p align="center">
  <img src="charts/eda_07_bivariate_correlations.png" width="100%" />
</p>

#### Multivariate Analysis — 3+ variables together

<p align="center">
  <img src="charts/eda_09_multi_contract_tenure.png" width="100%" />
</p>

**Contract x Tenure x Churn:** Month-to-month in first year = 55% churn. Two-year contracts = near-zero churn at ALL tenure levels.

<p align="center">
  <img src="charts/eda_10_multi_internet_services.png" width="100%" />
</p>

**Services x Internet x Churn:** Fiber + 0 add-ons = 52% churn. Fiber + 5-6 add-ons = only 15%. Services make customers sticky.

<p align="center">
  <img src="charts/eda_12_multi_risk_profile.png" width="100%" />
</p>

**Composite Risk Score:** Score 0 = 2% churn (safest). Score 6 = 73% churn (riskiest). A 36x difference!

<p align="center">
  <img src="charts/eda_13_multi_interactions.png" width="100%" />
</p>

**Interaction Effects:** High charges increase churn for month-to-month customers but barely affect two-year contract customers.

---

### Phase 4-5: Feature Engineering

Created **29 new features** across 7 categories, expanding from 19 to 71 total features:

| Category | Count | Key Features | Business Logic |
|----------|-------|-------------|---------------|
| **Service-Based** | 6 | `n_addons`, `n_protection`, `has_streaming` | More services = less churn |
| **Account** | 5 | `is_auto_pay`, `is_month_to_month`, `paperless_echeck` | Commitment and friction level |
| **Tenure-Based** | 5 | `is_new`, `is_loyal`, `tenure_sq` | Non-linear churn decay |
| **Billing** | 5 | `charge_diff`, `charge_ratio`, `log_total_charges` | Price change detection |
| **Demographics** | 3 | `has_family`, `senior_single` | Family stability factor |
| **Interaction** | 4 | `tenure_x_charges`, `fiber_no_protect`, `mtm_and_new` | Combined risk effects |
| **Composite** | 1 | `risk_score` (0-6) | 6 risk factors in one number |

**Risk Score Impact:**

| Score | Churn Rate | Profile |
|-------|-----------|---------|
| 0 (safest) | ~2% | Two-year, auto-pay, protection, loyal |
| 3 (moderate) | ~25% | Mixed risk factors |
| 6 (riskiest) | ~73% | M2M, fiber, e-check, new, no protection |

---

### Phase 6: Model Building

5-Fold Stratified Cross-Validation for all models:

| Model | Type | AUC-ROC | How It Works |
|-------|------|---------|-------------|
| Logistic Regression | Linear baseline | ~0.912 | Sigmoid function, linear boundary |
| Random Forest | Bagging (500 trees) | ~0.912 | Parallel random trees, averaged |
| XGBoost | Gradient Boosting | ~0.914 | Sequential error correction, regularized |
| LightGBM | Gradient Boosting | ~0.914 | Leaf-wise growth, fastest |
| CatBoost | Gradient Boosting | ~0.916 | Ordered boosting, best out-of-box |

**Hyperparameter Tuning:** Optuna Bayesian optimization — 50 trials per model, automatically finds best parameters.

**Advanced Techniques:** Multi-seed averaging (5 seeds x 5 folds = 25 models), importance-based feature selection.

---

### Phase 7: Ensemble and Submission

6 ensemble methods tested:

| Method | Description |
|--------|------------|
| Simple Average | Equal weight to all models |
| Weighted Average | Weight by CV score |
| Rank Average | Average ranks instead of probabilities (most robust) |
| Optimized Weights | scipy.optimize finds best weights |
| Stacking | Meta-learner on OOF predictions |
| Hill Climbing | Greedy weight optimization |

---

## Feature Importance

Top 15 features from LightGBM — **11 out of 15 are engineered features:**

| Rank | Feature | Type | Importance |
|------|---------|------|-----------|
| 1 | TotalCharges | Original | 1190 |
| 2 | MonthlyCharges | Original | 959 |
| 3 | avg_monthly_charge | Engineered | 902 |
| 4 | tenure | Original | 753 |
| 5 | charge_diff | Engineered | 732 |
| 6 | charge_ratio | Engineered | 695 |
| 7 | tenure_x_charges | Engineered | 546 |
| 8 | log_total_charges | Engineered | 298 |
| 9 | risk_score | Engineered | 197 |
| 10 | log_monthly_charges | Engineered | 184 |
| 11 | tenure_years | Engineered | 161 |
| 12 | n_protection | Engineered | 122 |
| 13 | contract_risk | Engineered | 122 |
| 14 | PaperlessBilling | Original | 115 |
| 15 | Dependents | Original | 113 |

**Feature engineering was the key differentiator in this competition.**

---

## Results Progression

| Phase | Approach | AUC-ROC | Gain |
|-------|----------|---------|------|
| Baseline | Logistic Regression (raw features) | ~0.910 | — |
| + Feature Engineering | 29 new features | ~0.912 | +0.002 |
| + Gradient Boosting | XGBoost / LightGBM / CatBoost | ~0.914 | +0.002 |
| + Optuna Tuning | 50 trials per model | ~0.916 | +0.002 |
| + Ensemble | 6 methods combined | **0.91405** | Final |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| ML Models | scikit-learn, XGBoost, LightGBM, CatBoost |
| Tuning | Optuna |
| Ensemble | scipy.optimize, custom stacking |

---

## How to Reproduce

```bash
# Clone and setup
git clone https://github.com/ParveenSharma00/kaggle-churn-prediction.git
cd kaggle-churn-prediction
pip install -r requirements.txt

# Download data from Kaggle
kaggle competitions download -c playground-series-s6e3 -p data/

# Run notebooks in order: 01 -> 02 -> 03 -> 04 -> 05 -> 06
```

---

## Key Takeaways

1. **Feature engineering > model selection** — 11 of top 15 features were engineered
2. **Ensemble always helps** — combining diverse models beats any single model
3. **Domain knowledge matters** — understanding telecom churn guided the best features
4. **CV score matches LB score** — proper stratified CV gives reliable estimates
5. **Start simple** — Logistic Regression at 0.912 was surprisingly close to boosted 0.916
6. **EDA drives everything** — every good feature came from patterns found during EDA

---

## Competition Stats

| Metric | Value |
|--------|-------|
| Competition | Playground Series S6E3 |
| Participants | 5,000+ |
| My Score | 0.91405 AUC-ROC |
| My Rank | ~1400 / 5000+ |
| Percentile | Top 28% |
| Models Trained | 10+ |
| Features Used | 71 |
| Ensembles Tested | 6 |
| Notebooks | 6 |
| EDA Charts | 13 |

---

## Author

**Parveen Sharma** — Senior Data Analyst | ML Practitioner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Parveen%20Sharma-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/parveen-150)
[![GitHub](https://img.shields.io/badge/GitHub-ParveenSharma00-black?style=flat-square&logo=github)](https://github.com/ParveenSharma00)

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for hosting Playground Series competitions
- Original dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) by IBM

---

<p align="center"><b>If you found this useful, please give it a star!</b></p>
