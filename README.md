# 🌲 Tree-Based Methods — Decision Trees, Random Forest, Boosting & BART

> **Skills Demonstrated:** Decision Trees · Random Forest · Gradient Boosting (XGBoost) · Bagging · BART · Tree Pruning · Cross-Validation · Feature Importance · Regression & Classification Trees · Python · Scikit-learn

---

## 🎯 Project Overview

This project implements and compares **five tree-based ensemble methods** across four real-world business problems — from retail sales forecasting to insurance purchase prediction. It answers a core ML question:

> *"When should we use a single decision tree vs bagging vs random forests vs boosting vs BART?"*

Six exercises are covered across four datasets:

1. **Boston Dataset** — Random Forest hyperparameter analysis (n_estimators vs max_features)
2. **Carseats Dataset** — Full regression tree pipeline: fit → prune → bag → RF → BART
3. **OJ Dataset** — Classification tree: fit → cross-validate → prune → compare
4. **Hitters Dataset** — Boosting for salary prediction with shrinkage tuning
5. **Caravan Dataset** — Boosting for insurance purchase prediction vs KNN & Logistic Regression
6. **Weekly Dataset** — Full benchmark: Boosting vs Bagging vs RF vs Logistic Regression

---

## 📁 Datasets Used

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| **Boston** | U.S. Census Bureau (Real) | 506 rows, 13 features | Housing price regression |
| **Carseats** | ISLP Simulated | 400 rows, 11 features | Sales prediction (regression) |
| **OJ** | Sales scan data (Real) | 1,070 rows, 18 features | Brand purchase classification |
| **Hitters** | MLB Baseball stats (Real) | 263 rows, 19 features | Salary prediction (regression) |
| **Caravan** | Dutch insurance (Real) | 5,822 rows, 86 features | Purchase prediction (classification) |
| **Weekly** | S&P 500 returns (Real) | 1,089 rows, 9 features | Market direction classification |

---

## 🔧 Techniques & Tools Applied

| Technique | Library | Purpose |
|-----------|---------|---------|
| Decision Tree (Regression & Classification) | `sklearn.tree` | Baseline interpretable model |
| Cost-Complexity Pruning | `sklearn` + `GridSearchCV` | Optimal tree size via CV |
| Bagging | `RandomForestRegressor(max_features=p)` | Variance reduction |
| **Random Forest** | `sklearn.ensemble.RandomForestRegressor` | Feature decorrelation + variance reduction |
| **Gradient Boosting** | `sklearn.ensemble.GradientBoostingRegressor/Classifier` | Sequential error correction |
| **BART** | `ISLP.bart.BART` | Bayesian Additive Regression Trees |
| Feature Importance | `.feature_importances_` | Identifying key predictors |
| K-Fold Cross-Validation | `sklearn.model_selection` | Model selection & tree size tuning |

**Libraries:** `numpy` · `pandas` · `scikit-learn` · `ISLP` · `matplotlib` · `statsmodels`

---

## 📊 Key Results

### Exercise 7 — Random Forest: Trees vs Features (Boston Dataset)

**Test MSE across n_estimators (10–590) and max_features settings:**

| max_features | Behavior | Convergence MSE |
|---|---|---|
| m = p (all features = Bagging) | Highest MSE, slowest convergence | ~13–15 |
| m = p/2 | Mid-range — good performance | ~11–12 |
| **m = √p** | **Lowest MSE, fastest convergence** | **~10–11** |

> **Finding:** Using m = √p features per split gives the best test MSE. More trees consistently reduce error until convergence (~200 trees). Beyond ~300 trees, additional estimators give diminishing returns.

---

### Exercise 8 — Carseats Sales Prediction (Full Pipeline)

**Train/Test Split:** 50/50 (random_state=30)

#### Progressive Model Improvement:

| Model | Test MSE | Improvement vs Baseline |
|-------|----------|------------------------|
| Full Regression Tree (depth=13) | **5.579** | Baseline |
| **Pruned Tree** (α=0.101, CV-selected) | **5.052** | ✅ 9.4% better |
| Bagging (max_features=p) | **2.615** | ✅ **53% better** |
| Random Forest (m=p) | **2.571** | ✅ **54% better** |
| Random Forest (m=p/2) | 2.758 | ✅ 51% better |
| Random Forest (m=√p) | 3.048 | ✅ 45% better |
| **BART** | **1.302** | ✅ **🏆 77% better** |

**Top Feature Importances (Bagging & Random Forest — consistent):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | **Price** | **0.292 – 0.297** |
| 2 | **ShelveLoc[Good]** | **0.211 – 0.214** |
| 3 | **Income** | **0.101 – 0.103** |
| 4 | Age | 0.089 – 0.099 |
| 5 | CompPrice | 0.096 – 0.100 |

> **Key Finding:** BART achieved test MSE of **1.302** — cutting the full tree error by 77%. Price and shelf location are the dominant sales drivers, consistent across all ensemble methods.

---

### Exercise 9 — OJ Orange Juice Purchase Classification

**Train/Test Split:** 800 / 270 observations

| Model | Training Error | Test Error | Notes |
|-------|---------------|------------|-------|
| Full Tree (171 terminal nodes) | **0.87%** | **22.22%** | ❌ Severely overfit |
| **Pruned Tree (depth=3, CV-selected)** | **17.75%** | **🏆 18.89%** | ✅ Best generalization |

**Cross-Validation Result:** Optimal tree size = **3 terminal nodes** (10-fold CV)

**Full Tree Text Summary (first terminal node):**
```
LoyalCH ≤ 0.04 AND StoreID ≤ 2.50 AND WeekofPurchase ≤ 267.50 → Predict MM (2 votes)
```

> **Key Finding:** Pruning from 171 nodes to just 3 reduced test error from 22.22% to 18.89% — a **15% improvement** — while dramatically improving interpretability. The most important split variable was `LoyalCH` (customer loyalty to Citrus Hill brand).

---

### Exercise 10 — Baseball Salary Prediction with Boosting (Hitters Dataset)

**Setup:** Log-transformed salary, first 200 obs = train, remainder = test

**Optimal Boosting parameters:** λ = 0.0257, n_estimators = 1,000

#### Model Comparison on Test Set:

| Model | Test MSE | vs Boosting |
|-------|----------|-------------|
| Multiple Linear Regression | 0.4526 | 2.06× worse |
| Ridge Regression | 0.4541 | 2.06× worse |
| **Gradient Boosting (optimal λ)** | **0.2202** | **🏆 Best** |
| Bagging | 0.2502 | 14% worse than boosting |

**Top Feature Importances (Boosting):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | **CAtBat** (career at-bats) | **0.436** |
| 2 | **CHits** (career hits) | **0.169** |
| 3 | AtBat (season at-bats) | 0.063 |
| 4 | CHmRun (career home runs) | 0.062 |
| 5 | RBI | 0.042 |

> **Key Finding:** Boosting reduced salary prediction error by **51%** vs linear regression. Career statistics (CAtBat, CHits) are far more predictive than single-season stats — seniority and consistency matter most in MLB salary determination.

---

### Exercise 11 — Caravan Insurance Purchase Prediction

**Setup:** 1,000 train / 4,822 test | Threshold = 20% probability

| Model | Precision (among predicted buyers) | Notes |
|-------|-------------------------------------|-------|
| Gradient Boosting | 16.95% | (49 true buyers / 289 predicted) |
| Logistic Regression | 18.69% | (54 true buyers / 289 predicted) |
| **KNN (K=5)** | **19.46%** | **(36 true buyers / 185 predicted) 🏆** |

**Top 3 Predictors (Boosting):**

| Feature | Importance | Description |
|---------|-----------|-------------|
| **MGODOV** | **0.0618** | Religious affiliation |
| **PPERSAUT** | **0.0601** | Car insurance policies |
| **AFIETS** | **0.0567** | Bicycle ownership |

> **Finding:** KNN (K=5) achieves the highest precision at 20% threshold — 1 in 5 predicted buyers actually converts. In insurance marketing, precision matters more than recall — targeting the right 185 people is more cost-effective than mass outreach.

---

### Exercise 12 — Full Model Benchmark: Weekly Market Direction

**Setup:** Weekly S&P 500 data, 70/30 train/test split

| Model | Test Error | Accuracy |
|-------|-----------|----------|
| **Logistic Regression** | **42.51%** | **57.49% 🏆** |
| Gradient Boosting | 43.43% | 56.57% |
| Bagging | 44.04% | 55.96% |
| Random Forest (√p) | 44.95% | 55.05% |

> **Key Finding:** All models perform near random chance (~50%) on stock market direction — confirming the Efficient Market Hypothesis. Logistic regression marginally wins, suggesting the signal is weak and linear. Complex ensemble methods do not add value when the underlying signal is near-zero.

---

## 💡 Business Insights

1. **BART Dominates Sales Forecasting:** BART reduced Carseats prediction error by 77% vs a single tree (MSE: 1.30 vs 5.58). For retail demand forecasting, BART is the top performer and should be the default choice over simpler ensemble methods.

2. **Pruning Prevents Costly Errors:** The OJ classification tree with 171 nodes had 22% test error. Pruning to 3 nodes reduced this to 19% — simpler models generalize better and are also more explainable to business stakeholders.

3. **Career Stats Beat Season Stats for Salary:** In MLB salary prediction, CAtBat (career at-bats) with importance 0.436 dwarfs all other features. HR teams should weight career consistency over single-season performance in compensation models.

4. **Precision > Recall in Insurance Marketing:** At a 20% probability threshold, KNN achieves 19.5% precision on Caravan insurance — meaning 1 in 5 targeted customers actually purchases. This is 3× better than random outreach (6% base rate), generating significant marketing ROI.

5. **Ensemble Methods Don't Fix Weak Signals:** When the true signal is near-zero (stock market returns), no ensemble method outperforms logistic regression. Model complexity should match signal strength — not default to the most complex option.

---

## 🗂️ File Structure

```
Chapter_8_Applied_Exercise_Solutions/
│
├── Chapter_8.ipynb          ← Main analysis notebook (all exercises)
├── Chapter_8.html           ← Rendered HTML version (easy browser viewing)
├── Chapter_8.qmd            ← Quarto source file
└── README.md                ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP scikit-learn pandas numpy matplotlib statsmodels

# Launch notebook
jupyter notebook Chapter_8.ipynb
```

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
*An Introduction to Statistical Learning with Applications in Python.* Springer.
Chapter 8: Tree-Based Methods — Applied Exercises 7–12.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM) provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shadalishah)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*
