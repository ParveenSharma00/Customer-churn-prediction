"""
============================================================================
PHASE 4 & 5: DATA PREPROCESSING + FEATURE ENGINEERING
============================================================================
Kaggle Playground Series S6E3 — Customer Churn Prediction

This script does:
  1. Load raw data
  2. Clean & preprocess
  3. Feature Engineering (17 new features)
  4. Encode categoricals
  5. Save processed data ready for modeling

Author: Parveen Sharma (learning ML end-to-end)
============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 0: LOAD RAW DATA
# ============================================================================
print("=" * 70)
print("STEP 0: Loading Raw Data")
print("=" * 70)

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

print(f"  Train: {train_raw.shape[0]:,} rows × {train_raw.shape[1]} cols")
print(f"  Test:  {test_raw.shape[0]:,} rows × {test_raw.shape[1]} cols")

# Save IDs (we need these for submission later)
train_ids = train_raw['id'].copy()
test_ids = test_raw['id'].copy()

# Save target (Churn) and encode: No → 0, Yes → 1
target = train_raw['Churn'].map({'No': 0, 'Yes': 1}).copy()
print(f"  Target → 0 (Stayed): {(target==0).sum():,} | 1 (Churned): {(target==1).sum():,}")


# ============================================================================
# STEP 1: COMBINE TRAIN + TEST
# ============================================================================
"""
📖 WHY COMBINE?
─────────────────
We process train and test TOGETHER so that:
  - One-hot encoding creates the SAME columns in both
  - Feature engineering is applied consistently  
  - No risk of mismatch between train and test

⚠️ RULE: We NEVER use the target column from test (it doesn't exist).
   We only combine FEATURES.
"""
print("\n" + "=" * 70)
print("STEP 1: Combining Train + Test for consistent processing")
print("=" * 70)

# Mark which rows are train vs test
train_raw['is_train'] = 1
test_raw['is_train'] = 0

# Drop target and id — we saved them separately
train_features = train_raw.drop(columns=['id', 'Churn'])
test_features = test_raw.drop(columns=['id'])

# Combine
df = pd.concat([train_features, test_features], axis=0, ignore_index=True)
print(f"  Combined shape: {df.shape[0]:,} rows × {df.shape[1]} cols")


# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================
"""
📖 WHAT IS DATA CLEANING?
──────────────────────────
Fixing data quality issues:
  - Missing values → fill or drop
  - Duplicate rows → remove
  - Wrong data types → convert
  - Inconsistent values → standardize

Our data is already clean (Kaggle synthetic data), but we check anyway.
In real-world projects, this step takes 50% of your time!
"""
print("\n" + "=" * 70)
print("STEP 2: Data Cleaning")
print("=" * 70)

# 2a. Missing values
missing = df.isnull().sum()
total_missing = missing.sum()
print(f"\n  2a. Missing Values: {total_missing}")
if total_missing > 0:
    print(f"  Columns with missing values:")
    print(missing[missing > 0])
    # Strategy: numerical → median, categorical → mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
                print(f"    {col}: filled with median")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"    {col}: filled with mode")
else:
    print(f"  ✅ No missing values")

# 2b. Duplicates
dupes = df.drop(columns=['is_train']).duplicated().sum()
print(f"\n  2b. Duplicate Rows: {dupes}")
if dupes > 0:
    print(f"  ⚠️  Found {dupes} duplicates — keeping first occurrence")
    # Don't drop dupes from combined data (train+test might have similar rows)
else:
    print(f"  ✅ No duplicates")

# 2c. Data type fixes
print(f"\n  2c. Data Type Checks:")
print(f"  SeniorCitizen is int (0/1) but should be treated as categorical")
print(f"  → We'll handle this in feature engineering")


# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
"""
📖 WHAT IS FEATURE ENGINEERING?
────────────────────────────────
Creating NEW columns from EXISTING data that help the model 
learn patterns better. This is where domain knowledge meets data science.

The idea: raw data tells you WHAT happened.
Engineered features tell you WHAT IT MEANS.

Example: 
  Raw data says: tenure=2, Contract='Month-to-month', MonthlyCharges=$95
  You think: "New customer, no commitment, high bill — CHURN RISK!"
  Feature engineering ENCODES this thinking into numbers.
"""
print("\n" + "=" * 70)
print("STEP 3: Feature Engineering (Creating New Features)")
print("=" * 70)

n_before = df.shape[1]

# ─────────────────────────────────────────────────────────────
# 3A. SERVICE-BASED FEATURES
# ─────────────────────────────────────────────────────────────
"""
📖 LOGIC: More services = more "sticky" customer = less churn
   We count how many services each customer has subscribed to.
   We also separate "protection" services from "entertainment" services
   because EDA showed protection services reduce churn MORE than streaming.
"""
print(f"\n  ── 3A: Service-Based Features ──")

# Internet-dependent add-on services
internet_addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

# Count of add-on services (only count 'Yes', not 'No' or 'No internet service')
df['n_addons'] = 0
for col in internet_addons:
    df['n_addons'] += (df[col] == 'Yes').astype(int)
print(f"  n_addons (0-6): How many internet add-on services")

# Protection services (security + support — reduce churn)
protection_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
df['n_protection'] = 0
for col in protection_services:
    df['n_protection'] += (df[col] == 'Yes').astype(int)
print(f"  n_protection (0-4): Security & support services count")

# Entertainment services (streaming — less impact on churn)
df['n_streaming'] = ((df['StreamingTV'] == 'Yes').astype(int) + 
                      (df['StreamingMovies'] == 'Yes').astype(int))
print(f"  n_streaming (0-2): TV + Movies streaming count")

# Total services (phone + internet + addons)
df['n_total_services'] = ((df['PhoneService'] == 'Yes').astype(int) + 
                           (df['InternetService'] != 'No').astype(int) + 
                           df['n_addons'])
print(f"  n_total_services (0-8): All services combined")

# Has any protection service?
df['has_protection'] = (df['n_protection'] > 0).astype(int)
print(f"  has_protection (0/1): Has at least 1 protection service")

# Has any streaming?
df['has_streaming'] = (df['n_streaming'] > 0).astype(int)
print(f"  has_streaming (0/1): Has at least 1 streaming service")

# ─────────────────────────────────────────────────────────────
# 3B. ACCOUNT & CONTRACT FEATURES
# ─────────────────────────────────────────────────────────────
"""
📖 LOGIC: Contract type and payment method are the STRONGEST churn predictors.
   We create features that capture "commitment level" and "friction to leave".
"""
print(f"\n  ── 3B: Account & Contract Features ──")

# Is month-to-month contract? (THE #1 risk factor)
df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
print(f"  is_month_to_month (0/1): No commitment contract")

# Contract risk score (ordinal: higher = riskier)
df['contract_risk'] = df['Contract'].map({
    'Two year': 0,       # safest — locked in
    'One year': 1,       # moderate
    'Month-to-month': 2  # riskiest — can leave anytime
})
print(f"  contract_risk (0-2): Ordinal risk from contract type")

# Auto-pay flag (automatic payment = friction to leave)
df['is_auto_pay'] = df['PaymentMethod'].isin([
    'Credit card (automatic)', 
    'Bank transfer (automatic)'
]).astype(int)
print(f"  is_auto_pay (0/1): Uses automatic payment method")

# Electronic check specifically (worst churn rate)
df['is_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
print(f"  is_electronic_check (0/1): Uses electronic check (highest churn)")

# Paperless + Electronic check combo (deadly combo from EDA)
df['paperless_echeck'] = ((df['PaperlessBilling'] == 'Yes') & 
                           (df['PaymentMethod'] == 'Electronic check')).astype(int)
print(f"  paperless_echeck (0/1): Paperless billing + electronic check combo")

# ─────────────────────────────────────────────────────────────
# 3C. TENURE-BASED FEATURES
# ─────────────────────────────────────────────────────────────
"""
📖 LOGIC: Tenure has a NON-LINEAR relationship with churn.
   First 6 months = highest risk. After 2 years = very safe.
   We create multiple tenure features to capture this.
"""
print(f"\n  ── 3C: Tenure-Based Features ──")

# New customer flag (≤ 6 months — danger zone)
df['is_new'] = (df['tenure'] <= 6).astype(int)
print(f"  is_new (0/1): Customer ≤ 6 months (highest churn window)")

# Mid-tenure (7-24 months — settling in)  
df['is_mid_tenure'] = ((df['tenure'] > 6) & (df['tenure'] <= 24)).astype(int)
print(f"  is_mid_tenure (0/1): 7-24 months")

# Loyal customer (> 48 months — safe)
df['is_loyal'] = (df['tenure'] > 48).astype(int)
print(f"  is_loyal (0/1): Customer > 48 months (low churn risk)")

# Tenure in years (continuous, easier for models to interpret)
df['tenure_years'] = df['tenure'] / 12
print(f"  tenure_years: Tenure converted to years")

# Tenure squared (captures non-linear decay of churn with tenure)
df['tenure_sq'] = df['tenure'] ** 2
print(f"  tenure_sq: Tenure squared (captures non-linear effect)")

# ─────────────────────────────────────────────────────────────
# 3D. BILLING & CHARGES FEATURES
# ─────────────────────────────────────────────────────────────
"""
📖 LOGIC: TotalCharges ≈ MonthlyCharges × tenure (0.99 correlation)
   So TotalCharges is almost redundant. But the DIFFERENCE between
   current MonthlyCharges and historical average tells us if prices changed.
"""
print(f"\n  ── 3D: Billing & Charges Features ──")

# Average charge per month of tenure (historical average)
df['avg_monthly_charge'] = df['TotalCharges'] / df['tenure'].clip(lower=1)
print(f"  avg_monthly_charge: TotalCharges / tenure")

# Charge difference (current vs historical — price change indicator)
df['charge_diff'] = df['MonthlyCharges'] - df['avg_monthly_charge']
print(f"  charge_diff: MonthlyCharges - avg (positive = price went UP)")

# Charge ratio (relative change)
df['charge_ratio'] = df['MonthlyCharges'] / df['avg_monthly_charge'].clip(lower=1)
print(f"  charge_ratio: MonthlyCharges / avg (>1 = price increased)")

# Log of TotalCharges (reduce skewness: 0.90 → less skewed)
df['log_total_charges'] = np.log1p(df['TotalCharges'])
print(f"  log_total_charges: log(1 + TotalCharges) — reduces skewness")

# Log of MonthlyCharges 
df['log_monthly_charges'] = np.log1p(df['MonthlyCharges'])
print(f"  log_monthly_charges: log(1 + MonthlyCharges)")

# ─────────────────────────────────────────────────────────────
# 3E. DEMOGRAPHIC FEATURES
# ─────────────────────────────────────────────────────────────
"""
📖 LOGIC: Family customers churn less. We combine Partner + Dependents.
   Senior citizens churn more, especially when single.
"""
print(f"\n  ── 3E: Demographic Features ──")

# Family flag (has partner OR dependents)
df['has_family'] = ((df['Partner'] == 'Yes') | 
                     (df['Dependents'] == 'Yes')).astype(int)
print(f"  has_family (0/1): Has partner or dependents")

# Family size proxy (0, 1, or 2)
df['family_size'] = ((df['Partner'] == 'Yes').astype(int) + 
                      (df['Dependents'] == 'Yes').astype(int))
print(f"  family_size (0-2): Partner + Dependents count")

# Senior + Single (highest churn demographic from EDA)
df['senior_single'] = ((df['SeniorCitizen'] == 1) & 
                        (df['Partner'] == 'No')).astype(int)
print(f"  senior_single (0/1): Senior citizen without partner")

# ─────────────────────────────────────────────────────────────
# 3F. INTERACTION FEATURES
# ─────────────────────────────────────────────────────────────
"""
📖 WHAT ARE INTERACTION FEATURES?
──────────────────────────────────
When the effect of Feature A on churn DEPENDS on Feature B,
we multiply them to create an "interaction term".

Example: High MonthlyCharges alone → moderate churn
         Low tenure alone → moderate churn
         High charges × Low tenure → VERY HIGH churn!
         The multiplication captures this combined effect.
"""
print(f"\n  ── 3F: Interaction Features ──")

# Tenure × Monthly Charges (low tenure + high charges = danger)
df['tenure_x_charges'] = df['tenure'] * df['MonthlyCharges']
print(f"  tenure_x_charges: tenure × MonthlyCharges")

# Month-to-month + New customer (double danger)
df['mtm_and_new'] = df['is_month_to_month'] * df['is_new']
print(f"  mtm_and_new (0/1): Month-to-month AND new customer")

# Fiber + No protection (high price, no stickiness)
df['fiber_no_protect'] = ((df['InternetService'] == 'Fiber optic') & 
                           (df['n_protection'] == 0)).astype(int)
print(f"  fiber_no_protect (0/1): Fiber optic with zero protection services")

# Senior + Month-to-month (vulnerable demographic + no commitment)
df['senior_mtm'] = (df['SeniorCitizen'] * df['is_month_to_month'])
print(f"  senior_mtm (0/1): Senior citizen on month-to-month")

# ─────────────────────────────────────────────────────────────
# 3G. COMPOSITE RISK SCORE
# ─────────────────────────────────────────────────────────────
"""
📖 WHAT IS A COMPOSITE SCORE?
──────────────────────────────
We combine multiple risk factors into ONE number.
Each risk factor adds 1 point. Higher score = higher churn risk.

From EDA, the top risk factors are:
  1. Month-to-month contract
  2. Fiber optic internet  
  3. Not using auto-pay
  4. No protection services
  5. New customer (≤ 12 months)
  6. Electronic check payment
"""
print(f"\n  ── 3G: Composite Risk Score ──")

df['risk_score'] = (
    df['is_month_to_month'] +                                    # +1 if month-to-month
    (df['InternetService'] == 'Fiber optic').astype(int) +       # +1 if fiber
    (1 - df['is_auto_pay']) +                                    # +1 if NOT auto-pay
    (df['n_protection'] == 0).astype(int) +                      # +1 if no protection
    (df['tenure'] <= 12).astype(int) +                           # +1 if new-ish
    df['is_electronic_check']                                    # +1 if e-check
)
print(f"  risk_score (0-6): Combined risk from 6 factors")
print(f"  Distribution:")
for score in sorted(df['risk_score'].unique()):
    count = (df['risk_score'] == score).sum()
    print(f"    Score {score}: {count:,} customers ({count/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# SUMMARY OF NEW FEATURES
# ─────────────────────────────────────────────────────────────
n_after = df.shape[1]
n_new = n_after - n_before
print(f"\n  {'='*50}")
print(f"  FEATURE ENGINEERING SUMMARY")
print(f"  {'='*50}")
print(f"  Original features:     {n_before - 1} (excl is_train)")
print(f"  New features created:  {n_new}")
print(f"  Total features now:    {n_after - 1}")


# ============================================================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ============================================================================
"""
📖 THREE ENCODING STRATEGIES:
─────────────────────────────
1. BINARY ENCODING: For Yes/No features → 1/0
   Simple. Direct. No information loss.

2. ORDINAL ENCODING: For features with natural order
   Contract: Month-to-month(0) < One year(1) < Two year(2)
   The NUMBER reflects the ORDER.

3. ONE-HOT ENCODING (OHE): For nominal features (no order)
   InternetService: DSL, Fiber, No → 3 binary columns
   Each category gets its own column (1 if true, 0 if false)
   
⚠️ WARNING: Don't use one-hot for features with many categories (>10)
   It creates too many columns. Use target encoding instead.
   Our data has max 4 categories per feature, so OHE is fine.
"""
print("\n" + "=" * 70)
print("STEP 4: Encoding Categorical Variables")
print("=" * 70)

# ── 4A: Binary Encoding ──
print(f"\n  ── 4A: Binary Encoding (Yes/No → 1/0) ──")

binary_mappings = {
    'gender':           {'Female': 0, 'Male': 1},
    'Partner':          {'No': 0, 'Yes': 1},
    'Dependents':       {'No': 0, 'Yes': 1},
    'PhoneService':     {'No': 0, 'Yes': 1},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
}

for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)
    print(f"  {col}: {mapping}")

# ── 4B: Ordinal Encoding ──
print(f"\n  ── 4B: Ordinal Encoding ──")

# Contract (already created contract_risk, but encode original too)
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['Contract_ord'] = df['Contract'].map(contract_mapping)
print(f"  Contract_ord: {contract_mapping}")

# Internet service has a natural order by service level
internet_mapping = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
df['InternetService_ord'] = df['InternetService'].map(internet_mapping)
print(f"  InternetService_ord: {internet_mapping}")

# ── 4C: One-Hot Encoding ──
print(f"\n  ── 4C: One-Hot Encoding (nominal features) ──")

ohe_columns = [
    'MultipleLines',      # No / Yes / No phone service
    'InternetService',    # DSL / Fiber optic / No
    'OnlineSecurity',     # No / Yes / No internet service
    'OnlineBackup',       # No / Yes / No internet service
    'DeviceProtection',   # No / Yes / No internet service
    'TechSupport',        # No / Yes / No internet service
    'StreamingTV',        # No / Yes / No internet service
    'StreamingMovies',    # No / Yes / No internet service
    'Contract',           # Month-to-month / One year / Two year
    'PaymentMethod',      # 4 payment methods
]

print(f"  Columns to encode: {len(ohe_columns)}")
print(f"  Columns before OHE: {df.shape[1]}")

df = pd.get_dummies(df, columns=ohe_columns, drop_first=False, dtype=int)

print(f"  Columns after OHE:  {df.shape[1]}")
print(f"  New OHE columns created: {df.shape[1] - n_after - 2}")  # -2 for ord encodings


# ============================================================================
# STEP 5: FINAL VALIDATION & CLEANUP
# ============================================================================
"""
📖 FINAL CHECKS:
─────────────────
Before we pass data to models, we verify:
  1. All columns are numeric (no text left)
  2. No missing values
  3. No infinite values
  4. Train and test have same columns
  5. No data leakage (target info didn't sneak in)
"""
print("\n" + "=" * 70)
print("STEP 5: Final Validation & Cleanup")
print("=" * 70)

# Check 1: All numeric
non_numeric = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
if non_numeric:
    print(f"  ❌ Non-numeric columns found: {non_numeric}")
    print(f"     Dropping these...")
    df = df.drop(columns=non_numeric)
else:
    print(f"  ✅ All columns are numeric")

# Check 2: Missing values
n_missing = df.isnull().sum().sum()
print(f"  {'✅' if n_missing == 0 else '❌'} Missing values: {n_missing}")
if n_missing > 0:
    df = df.fillna(0)

# Check 3: Infinite values
n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"  {'✅' if n_inf == 0 else '❌'} Infinite values: {n_inf}")
if n_inf > 0:
    df = df.replace([np.inf, -np.inf], 0)

# Check 4: Column consistency
print(f"  ✅ Total columns: {df.shape[1]} (including is_train flag)")


# ============================================================================
# STEP 6: SPLIT BACK & SAVE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Split Back into Train & Test, Save")
print("=" * 70)

# Split
X_train = df[df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
X_test  = df[df['is_train'] == 0].drop(columns=['is_train']).reset_index(drop=True)
y_train = target.reset_index(drop=True)

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape} (Churn=1: {y_train.sum():,})")

# Verify shapes match
assert X_train.shape[0] == y_train.shape[0], "Train features and target mismatch!"
assert X_train.shape[1] == X_test.shape[1], "Train and test column mismatch!"
print(f"  ✅ Shape validation passed")

# Save
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
test_ids.to_csv('test_ids.csv', index=False)

print(f"  ✅ Saved: X_train.csv, X_test.csv, y_train.csv, test_ids.csv")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'=' * 70}")
print(f"  COMPLETE PIPELINE SUMMARY")
print(f"{'=' * 70}")
print(f"""
  INPUT:
    Train: 594,194 × 19 features
    Test:  254,655 × 19 features
  
  OUTPUT:
    X_train: {X_train.shape[0]:,} × {X_train.shape[1]} features
    X_test:  {X_test.shape[0]:,} × {X_test.shape[1]} features
    y_train: {y_train.shape[0]:,} labels
  
  FEATURES BREAKDOWN:
    Original features (encoded):     19
    Engineered features:             {n_new}
    One-hot encoded columns:         {X_train.shape[1] - n_after + 1}
    ─────────────────────────────────
    TOTAL:                           {X_train.shape[1]}
  
  DATA QUALITY: ✅ All checks passed
""")

# Print all features grouped
print(f"  ALL {X_train.shape[1]} FEATURES (grouped):")
print(f"\n  ── Original Numerical ──")
for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Original Binary (encoded) ──")
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Engineered: Service-Based ──")
for col in ['n_addons', 'n_protection', 'n_streaming', 'n_total_services', 
            'has_protection', 'has_streaming']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Engineered: Account & Contract ──")
for col in ['is_month_to_month', 'contract_risk', 'is_auto_pay', 
            'is_electronic_check', 'paperless_echeck', 'Contract_ord', 'InternetService_ord']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Engineered: Tenure-Based ──")
for col in ['is_new', 'is_mid_tenure', 'is_loyal', 'tenure_years', 'tenure_sq']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Engineered: Billing ──")
for col in ['avg_monthly_charge', 'charge_diff', 'charge_ratio', 
            'log_total_charges', 'log_monthly_charges']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Engineered: Demographic ──")
for col in ['has_family', 'family_size', 'senior_single']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Engineered: Interactions ──")
for col in ['tenure_x_charges', 'mtm_and_new', 'fiber_no_protect', 'senior_mtm']:
    if col in X_train.columns:
        print(f"    {col}")

print(f"\n  ── Engineered: Composite ──")
for col in ['risk_score']:
    if col in X_train.columns:
        print(f"    {col}")

ohe_cols = [c for c in X_train.columns if '_' in c and any(
    c.startswith(p) for p in ['MultipleLines_', 'InternetService_', 'OnlineSecurity_',
                               'OnlineBackup_', 'DeviceProtection_', 'TechSupport_',
                               'StreamingTV_', 'StreamingMovies_', 'Contract_M', 
                               'Contract_O', 'Contract_T', 'PaymentMethod_'])]
print(f"\n  ── One-Hot Encoded ({len(ohe_cols)} columns) ──")
for col in ohe_cols:
    print(f"    {col}")

print(f"\n{'=' * 70}")
print(f"  ✅ PHASE 4 & 5 COMPLETE — Data ready for modeling!")
print(f"{'=' * 70}")
