# Data Files

## Download Instructions

The training data is too large for GitHub. Download from Kaggle:

**Competition:** [Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3/data)

### Required Files:
1. `train.csv` (~77 MB) — Training data with 594,194 rows
2. `test.csv` (~33 MB) — Test data with 254,655 rows
3. `sample_submission.csv` (~2 MB) — Submission format

### Optional (for Part 4: Advanced Techniques):
4. `WA_Fn-UseC_-Telco-Customer-Churn.csv` — [Original dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### How to Download:
```bash
# Using Kaggle CLI
kaggle competitions download -c playground-series-s6e3 -p data/
unzip data/playground-series-s6e3.zip -d data/
```

Or download manually from the competition page and place all CSV files in this folder.
