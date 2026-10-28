# Credit Card Fraud Detection — Real-Time Machine Learning System

**Author:** Emmanuel Odigie Okohe  
**ORCID:** 0009-0006-0897-707X  
**MSc Artificial Intelligence** — Sheffield Hallam University  
**GitHub:** [github.com/Odigie43](https://github.com/Odigie43)

---

## Overview

This project implements a real-time credit card fraud detection system using supervised machine learning. The system ingests raw transactional data, trains four classification models with SMOTE-based class imbalance correction, and exposes a live prediction interface that returns a fraud probability score for any submitted transaction — enabling point-of-authorisation fraud decisions rather than post-settlement batch review.

The deployed application is built from `Clement_ML.ipynb` and provides an end-to-end pipeline covering exploratory data analysis, model training, comparative evaluation, and live inference.

---

## Dataset

| Metric | Value |
|---|---|
| Total transactions | 339,607 |
| Fraud cases | 1,782 (0.52%) |
| Class imbalance handling | SMOTE (Synthetic Minority Oversampling Technique) |

The dataset exhibits severe class imbalance — a realistic condition in production fraud detection environments — making SMOTE balancing a necessary preprocessing step to prevent model bias toward the majority (non-fraud) class.

---

## Models Trained

All four models are trained in a single pipeline. Results are cached so training runs only once per session.

| Model | AUC-ROC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.9072 | 0.1600 | 0.1205 | 0.0656 | 0.7416 |
| Decision Tree | 0.9707 | 0.3632 | 0.2015 | 0.1131 | **0.9242** |
| Random Forest | 0.9845 | 0.7541 | 0.4449 | 0.3005 | 0.8567 |
| **XGBoost** | **0.9959** | **0.8683** | **0.6041** | **0.4514** | 0.9129 |

**XGBoost is the best-performing model** across the three most operationally significant metrics: AUC-ROC, PR-AUC, and F1. An AUC-ROC of 0.9959 indicates near-perfect discrimination between fraudulent and legitimate transactions across all classification thresholds.

PR-AUC (Precision-Recall AUC) is the primary evaluation metric for this problem domain because the dataset is highly imbalanced. A high PR-AUC (0.8683 for XGBoost) demonstrates that the model maintains strong precision even as recall increases — the characteristic that matters most in a production fraud flag system where false positives carry significant customer experience cost.

---

## Application Pipeline

The Streamlit application exposes five sequential stages:

**Step 1 — Data Upload**  
Upload `credit_card_frauds.csv` via the sidebar to initialise the pipeline.

**Step 2 — Exploratory Data Analysis**  
Visualise class imbalance distribution, transaction amount distributions, hourly fraud occurrence patterns, and feature correlation matrices.

**Step 3 — Model Training**  
All four models (Logistic Regression, Decision Tree, Random Forest, XGBoost) are trained with SMOTE balancing applied to the training split. Training results are cached to avoid recomputation on page reload.

**Step 4 — Evaluation**  
Full comparative evaluation suite: ROC curves, Precision-Recall curves, confusion matrices, feature importance charts, and threshold optimisation analysis — all generated directly from the trained models.

**Step 5 — Live Predictor**  
Enter the details of any transaction and receive a real-time fraud probability score from the trained XGBoost model. This stage demonstrates the system's capability for point-of-authorisation inference.

---

## Technical Architecture

- **Language:** Python 3
- **ML Framework:** scikit-learn, XGBoost
- **Imbalance Handling:** imbalanced-learn (SMOTE)
- **Application Layer:** Streamlit
- **Evaluation Metrics:** AUC-ROC, PR-AUC, F1, Precision, Recall
- **Notebook Source:** `Clement_ML.ipynb`

---

## Research Context

This system is part of a broader programme of independently published AI research by the author. The accompanying research paper — *Real-Time Credit Card Fraud Detection Using Machine Learning* — is published on Zenodo and indexed under ORCID 0009-0006-0897-707X. The paper provides a full account of the feature engineering methodology, model selection rationale, and architectural decisions reflected in this implementation.

The 0.9959 AUC-ROC achieved by XGBoost on a 339,607-transaction dataset with 0.52% fraud prevalence places this system's discriminative performance in the top tier of published fraud detection benchmarks for comparable dataset characteristics.

---

## How to Run

```bash
git clone https://github.com/Odigie43/Credit_card_fraud_detection_real_time.git
cd Credit_card_fraud_detection_real_time
pip install -r requirements.txt
streamlit run app.py
```

Upload `credit_card_frauds.csv` in the sidebar when the application loads.

---

## Licence

This project is released for research and educational purposes. Please cite the accompanying Zenodo publication if you use this work in your own research.

---

*Emmanuel Odigie Okohe · ORCID 0009-0006-0897-707X · github.com/Odigie43*
