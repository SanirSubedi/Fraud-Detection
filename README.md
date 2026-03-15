# Fraud Detection System — Engineering Documentation

### Technical Report — Machine Learning Course Project

**Author:** Sanir Subedi , Ajit GC , Abhishek Kc , Aayush Shrestha , Taqi Muhammad 
**Course:** Machine Learning  
**Submission Date:** March 2026  
**Repository:** https://github.com/SanirSubedi/Fraud-Detection  
**Dataset Source:** Kaggle — Fraud Detection Dataset by Aman Ali Siddiqui  
https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Data Exploration and Observations](#3-data-exploration-and-observations)
4. [Data Cleaning and Preprocessing](#4-data-cleaning-and-preprocessing)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Development](#6-model-development)
7. [Results and Evaluation](#7-results-and-evaluation)
8. [Best Model Selection](#8-best-model-selection)
9. [Deployment](#9-deployment)
10. [Conclusions](#10-conclusions)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Background

Financial fraud is a persistent and damaging problem. Fraudulent transactions cost the global economy tens of billions of dollars annually. Detecting fraud manually is not feasible at the scale of modern digital payments, where millions of transactions are processed every hour. Machine learning offers a practical path forward — a trained model can evaluate each transaction in milliseconds and flag suspicious activity automatically.

This project builds and evaluates a machine learning system for binary fraud classification. Given a set of transaction features, the system predicts whether a transaction is fraudulent (1) or legitimate (0).

### 1.2 Project Objectives

- Load and explore a large real-world financial transaction dataset
- Identify patterns and signals that distinguish fraud from legitimate activity
- Clean and engineer features that improve model performance
- Train and compare four machine learning models
- Select the best-performing model using appropriate evaluation metrics
- Deploy the trained model for live inference

### 1.3 Why This Dataset

The PaySim fraud dataset was chosen because it simulates real mobile money transactions and includes a ground truth label (`isFraud`), making it directly suitable for supervised classification. The dataset is large enough — over six million records — to reflect the scale seen in production financial systems. It also replicates a core real-world challenge: fraud cases are extremely rare, making up only 0.13% of all transactions. This imbalance is not a simplified classroom condition — it is the exact difficulty that production fraud detection systems face every day.

---

## 2. Dataset

### 2.1 Source

| Property | Detail |
|---|---|
| Name | Fraud Detection Dataset |
| Author | Aman Ali Siddiqui |
| Platform | Kaggle |
| URL | https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset |
| File | AIML Dataset.csv |

### 2.2 Overview

| Property | Value |
|---|---|
| Total rows | 6,362,620 |
| Total columns | 11 |
| Task type | Binary classification |
| Fraud rows | 8,213 |
| Legitimate rows | 6,354,407 |
| Missing values | None |

### 2.3 Column Schema

| Column | Type | Description |
|---|---|---|
| `step` | Integer | Simulated time unit. 1 step = 1 hour of simulation |
| `type` | String | Transaction type: PAYMENT, TRANSFER, CASH_OUT, DEPOSIT, DEBIT |
| `amount` | Float | Amount transacted |
| `nameOrig` | String | Sender account identifier |
| `oldbalanceOrg` | Float | Sender balance before the transaction |
| `newbalanceOrig` | Float | Sender balance after the transaction |
| `nameDest` | String | Receiver account identifier |
| `oldbalanceDest` | Float | Receiver balance before the transaction |
| `newbalanceDest` | Float | Receiver balance after the transaction |
| `isFlaggedFraud` | Integer | Rule-based system flag |
| `isFraud` | Integer | **Target label** — 1 = fraud, 0 = legitimate |

### 2.4 Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 6,354,407 | 99.87% |
| Fraud (1) | 8,213 | 0.13% |

The fraud-to-legitimate ratio is approximately 1:774. A model that predicts every transaction as legitimate would achieve 99.87% accuracy while detecting zero fraud. This makes accuracy an unreliable evaluation metric. Precision, Recall and F1-Score on the fraud class are used throughout this report instead.

---

## 3. Data Exploration and Observations

Exploratory data analysis was carried out before any modelling work. The findings below directly shaped both feature selection and model choice.

### 3.1 Fraud is Restricted to Two Transaction Types

The most significant finding from the EDA was that fraud occurs exclusively in TRANSFER and CASH_OUT transactions. No fraud was found in PAYMENT, DEPOSIT or DEBIT transactions across the full 6.3 million row dataset. This makes `type` the single most discriminating feature available.

### 3.2 Account Drain Pattern

A filter was applied to identify transactions where the sender had a positive balance that dropped to exactly zero:

```python
zero_after_transfer = df[
    (df["oldbalanceOrg"] > 0) &
    (df["newbalanceOrig"] == 0) &
    (df["type"].isin(["TRANSFER", "CASH_OUT"]))
]
# Result: 1,188,074 transactions
```

This pattern — a sender's account being fully emptied — is a strong behavioural indicator of fraud and directly motivated the `balanceDiffOrig` feature described in Section 5.

### 3.3 Single-Use Fraudulent Accounts

Every fraudulent sender account (`nameOrig`) appeared exactly once in the dataset. No fraud sender made more than one transaction. This is consistent with stolen or disposable accounts used for a single transfer and then abandoned. It also confirms that dropping `nameOrig` as a feature is correct — unique identifiers that never repeat cannot generalise to new transactions.

### 3.4 Near-Perfect Correlation Between Balance Columns

A correlation analysis showed that `oldbalanceOrg` and `newbalanceOrig` had a Pearson correlation coefficient of 0.998. These two columns are nearly identical in information content. The difference between them is a far more useful signal than either column individually.

| Feature Pair | Pearson Correlation |
|---|---|
| `oldbalanceOrg` ↔ `newbalanceOrig` | 0.998 |
| `oldbalanceDest` ↔ `newbalanceDest` | moderate positive |
| `amount` ↔ `isFraud` | weak positive |

### 3.5 Transaction Amount Distribution

Transaction amounts are heavily right-skewed. A log transformation (`np.log1p`) was applied for visualisation. Fraud transactions tend to involve larger amounts on average than legitimate transactions, which was confirmed by a boxplot comparing the two classes.

### 3.6 Fraud Frequency Over Time

Plotting fraud count by time step showed that fraud occurs in bursts rather than at a constant rate. This suggests coordinated fraud activity. The `step` column was dropped before modelling because time-based patterns from simulated data do not generalise to live transaction streams.

---

## 4. Data Cleaning and Preprocessing

### 4.1 Missing Values

```python
df.isnull().sum()  # Result: 0 across all columns
```

No missing values were found. No imputation was necessary.

### 4.2 Column Removal

| Column | Reason for Removal |
|---|---|
| `step` | Raw time counter — not a generalisable predictive feature |
| `nameOrig` | Unique per transaction — cannot generalise to unseen accounts |
| `nameDest` | Same reason as `nameOrig` |
| `isFlaggedFraud` | Near-constant — effectively all zeros, adds no signal |

```python
df.drop(columns="step", inplace=True)
df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)
```

### 4.3 Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` is the critical parameter here. On a 0.13% minority class, a purely random split could accidentally concentrate nearly all fraud cases in one partition. Stratification guarantees the fraud ratio is identical in both splits.

| Split | Rows | Fraud Cases | Fraud % |
|---|---|---|---|
| Training | 5,090,096 | 6,570 | 0.13% |
| Testing | 1,272,524 | 1,643 | 0.13% |

`random_state=42` ensures the split is reproducible across runs.

### 4.4 Preprocessing Pipeline

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ]
)
```

`OneHotEncoder` converts the `type` column into five binary columns, one per transaction type. Machine learning models cannot process string values directly.

`StandardScaler` rescales all numeric features to zero mean and unit variance. Without scaling, high-magnitude columns like `amount` would numerically dominate low-magnitude columns in distance-based calculations.

`handle_unknown="ignore"` prevents the pipeline from raising an error if an unseen transaction type appears at inference time.

All preprocessing steps were wrapped inside a scikit-learn `Pipeline` with each classifier. This prevents data leakage and ensures inference automatically applies the same transformations used during training.

---

## 5. Feature Engineering

Two new features were derived from the existing balance columns:

```python
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
```

| Feature | Formula | Rationale |
|---|---|---|
| `balanceDiffOrig` | `oldbalanceOrg − newbalanceOrig` | Direct measure of money leaving the sender. More informative than either raw balance column given their 0.998 correlation |
| `balanceDiffDest` | `newbalanceDest − oldbalanceDest` | Measures money arriving at the receiver. A large discrepancy between this and `balanceDiffOrig` can indicate irregular routing |

### Final Feature Set Used for Modelling

| Feature | Type | Origin |
|---|---|---|
| `type` | Categorical | Original |
| `amount` | Numeric | Original |
| `oldbalanceOrg` | Numeric | Original |
| `newbalanceOrig` | Numeric | Original |
| `oldbalanceDest` | Numeric | Original |
| `newbalanceDest` | Numeric | Original |
| `balanceDiffOrig` | Numeric | Engineered |
| `balanceDiffDest` | Numeric | Engineered |

**Target:** `isFraud` (0 = legitimate, 1 = fraud)

---

## 6. Model Development

Four models were trained and evaluated. Each was wrapped in a scikit-learn `Pipeline` to ensure consistent preprocessing across training and inference.

### 6.1 Logistic Regression

Logistic Regression estimates the probability of class membership using a linear decision boundary. The `class_weight='balanced'` parameter re-weights the loss function to penalise misclassification of the minority fraud class more heavily.

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])
```

**Known limitation:** The linear decision boundary cannot capture the conditional, rule-based nature of fraud in this dataset — for example, fraud requires both a specific transaction type AND a specific balance behaviour simultaneously.

### 6.2 Linear Regression

Linear Regression is a regression model and not a classifier. It was included to demonstrate why regression approaches are inappropriate for binary classification. The continuous prediction output was converted to class labels using a 0.5 threshold:

```python
y_pred_lin_raw = lin_pipeline.predict(X_test)
y_pred_lin = (y_pred_lin_raw >= 0.5).astype(int)
```

R² and Mean Squared Error were recorded alongside classification metrics to characterise its regression fit.

**Known limitation:** Linear Regression has no concept of class boundaries or probability calibration, and no mechanism to handle class imbalance. Its inclusion is comparative and pedagogical.

### 6.3 K-Nearest Neighbors

KNN classifies each transaction by finding the k most similar transactions in the training set and assigning the majority label among those neighbours. It makes no assumptions about the underlying data distribution.

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ))
])
```

Due to KNN's O(n) prediction complexity, training on the full 5 million row set would be computationally prohibitive for a course project. A stratified 100,000-row sample was used for training. The test set remained the full 1,272,524 rows.

```python
X_train_sample = X_train.sample(n=100_000, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
```

The sample preserved the 0.13% fraud ratio, ensuring the model trained on representative fraud examples despite the reduced size.

### 6.4 Decision Tree

Decision Tree builds a hierarchy of if-then rules by recursively splitting the feature space to maximise class separation. `max_depth=10` prevents the tree from memorising the training data.

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ))
])
```

---

## 7. Results and Evaluation

### 7.1 Evaluation Metrics

Three metrics were prioritised over accuracy:

**Precision** — of all transactions flagged as fraud, the fraction that are genuine fraud. High precision reduces false alarms and minimises unjustified account blocks for legitimate customers.

**Recall** — of all actual fraud transactions, the fraction the model caught. High recall minimises missed fraud.

**F1-Score** — the harmonic mean of precision and recall. This is the primary selection metric because it forces both precision and recall to be meaningfully high simultaneously.

### 7.2 Full Classification Reports

#### Logistic Regression

```
              precision    recall  f1-score   support

   Not Fraud       1.00      0.95      0.97   1270881
       Fraud       0.02      0.94      0.04      1643

    accuracy                           0.95   1272524
   macro avg       0.51      0.95      0.51   1272524
weighted avg       1.00      0.95      0.97   1272524
```

#### Linear Regression (threshold = 0.5)

Linear Regression produced the weakest results of all four models. Its continuous output is not calibrated for binary classification and it has no mechanism to handle imbalanced classes. Precision and Recall on the fraud class were both near zero.

#### K-Nearest Neighbors

```
              precision    recall  f1-score   support

   Not Fraud       1.00      1.00      1.00   1270881
       Fraud       0.93      0.57      0.71      1643

    accuracy                           1.00   1272524
   macro avg       0.96      0.78      0.85   1272524
weighted avg       1.00      1.00      1.00   1272524
```

#### Decision Tree

```
              precision    recall  f1-score   support

   Not Fraud       1.00      0.99      1.00   1270881
       Fraud       0.17      0.99      0.29      1643

    accuracy                           0.99   1272524
   macro avg       0.59      0.99      0.65   1272524
weighted avg       1.00      0.99      1.00   1272524
```

### 7.3 Comparison Table

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|---|---|---|---|---|
| Logistic Regression | 94.73% | 2.26% | 94.34% | 4.42% |
| Linear Regression | — | ~0% | ~0% | ~0% |
| **KNN (k=5)** | **99.94%** | **92.58%** | **56.97%** | **70.54%** |
| Decision Tree | 99.39% | 17.34% | 98.66% | 29.50% |

### 7.4 Analysis

**Logistic Regression** achieved 94% Recall but only 2% Precision. The model compensated for the imbalance by shifting its decision threshold so aggressively that it flagged the majority of transactions as fraud. In a deployed system, this produces approximately 50 false alarms for every genuine fraud detected — operationally unworkable.

**Linear Regression** confirmed that regression is not suitable for binary classification. The result validates the decision to include it as a comparison point only.

**Decision Tree** reached 99% Recall but at 17% Precision — for every genuine fraud caught, roughly five legitimate transactions were incorrectly blocked. The balanced class weight caused excessive over-flagging despite the tree's rule-learning strengths.

**KNN** returned the highest F1-Score of 70.54% with 92.58% Precision. A fraud analyst reviewing KNN alerts would find genuine fraud in approximately 9 out of 10 cases. Its Recall of 57% means some fraud is missed, but the trade-off is substantially more practical than the alternatives.

---

## 8. Best Model Selection

### 8.1 Selected Model

**K-Nearest Neighbors — k=5, trained on 100,000-row stratified sample**

F1-Score: **70.54%** | Precision: **92.58%** | Recall: **56.97%**

### 8.2 Justification

KNN was selected for three reasons.

First, it achieved the highest F1-Score among all four models. F1 cannot be inflated by sacrificing either precision or recall — both must be meaningfully high simultaneously.

Second, its Precision of 92.58% has direct operational significance. At this precision level, fraud alerts are almost always actionable. Logistic Regression's 2.26% Precision means 97 out of every 100 fraud alerts would be false — the system would lose analyst trust and become ignored in practice.

Third, KNN's non-parametric nature is well-suited to this dataset. Fraud transactions share a specific feature fingerprint: TRANSFER or CASH_OUT type, sender balance dropping to zero, and often large amounts. These cases cluster tightly together in feature space. KNN's nearest-neighbour mechanism exploits this clustering without needing explicitly programmed rules.

### 8.3 Model Persistence

```python
joblib.dump(knn_pipeline, "frad_detection_pipeline.pkl")
```

The complete pipeline — the `ColumnTransformer` preprocessor and the trained KNN classifier — was serialised as a single file. Loading this file at inference time reproduces the exact preprocessing and classification behaviour from training with no additional setup.

---

## 9. Deployment

### 9.1 Streamlit Web Application

A browser-based prediction interface was developed using Streamlit (`app.py`). The application loads the persisted pipeline and presents a form where users can enter transaction details and receive a fraud prediction in real time.


### 9.2 Inference Pipeline

When a transaction is submitted, the following steps execute inside the loaded pipeline:

1. Input is wrapped into a single-row pandas DataFrame
2. `OneHotEncoder` converts `type` to binary columns
3. `StandardScaler` rescales all numeric values using training set statistics
4. KNN identifies the 5 nearest neighbours in the stored 100k training sample
5. Majority vote returns 0 (legitimate) or 1 (fraud)

The entire process completes in under one second per transaction.

---

## 10. Conclusions

### 10.1 Summary

This project built a complete fraud detection pipeline — from raw data loading through exploratory analysis, feature engineering, model training, evaluation and deployment. Four models were trained and compared on a dataset of 6.3 million transactions with a 0.13% fraud rate. KNN was selected as the best model with a 70.54% F1-Score on the fraud class.

### 10.2 Key Findings

Fraud in this dataset is confined entirely to TRANSFER and CASH_OUT transaction types. Transaction type is therefore the single most discriminating available feature.

The account drain pattern — sender balance dropping to exactly zero — is a reliable fraud indicator that appeared in over 1.1 million transactions and was effectively captured by the `balanceDiffOrig` engineered feature.

Accuracy is not a valid metric for this problem. All four models exceeded 94% accuracy, yet only KNN demonstrated genuinely useful fraud detection. F1-Score was the metric that separated the models meaningfully.

Precision is operationally as important as Recall. High Recall with poor Precision floods analysts with false alerts and blocks legitimate customers — a practical failure regardless of the statistical numbers reported.

### 10.3 Limitations

KNN was trained on a 100,000-row sample due to the computational cost of the full training set. Training on the complete data using an approximate nearest-neighbour library would likely improve Recall without sacrificing Precision.

No decision threshold tuning was performed. The default 0.5 threshold was used throughout. In production, the threshold would be calibrated against the relative cost of a missed fraud versus a false alarm.

The model does not incorporate account-level history or network features. The observation that some receiver accounts appeared in over 100 transactions — a potential money mule indicator — was not exploited in the current feature set.

### 10.4 Future Work

| Priority | Improvement | Expected Benefit |
|---|---|---|
| High | Replace KNN with Random Forest or XGBoost | Higher F1, better generalisation, full dataset training |
| High | Apply SMOTE oversampling | Improved Recall on fraud class |
| Medium | Tune decision threshold via cost matrix | Optimised precision-recall trade-off for deployment |
| Medium | Train KNN on full dataset using FAISS | Remove sampling limitation |
| Low | Integrate MLflow experiment tracking | Reproducibility and model versioning |

---

## 11. References

Siddiqui, A. A. (n.d.). *Fraud Detection Dataset*. Kaggle.
Retrieved from https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *The 28th European Modeling and Simulation Symposium, EMSS*, Larnaca, Cyprus.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

Cover, T., & Hart, P. (1967). Nearest neighbour pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.

---

*Submitted in partial fulfilment of the Machine Learning course requirements.*  
*All analysis and implementation conducted independently.*  
*Dataset accessed via Kaggle, referenced in Section 2.1.*