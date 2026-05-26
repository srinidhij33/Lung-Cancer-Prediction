# 🫁 Lung Cancer Prediction & NLP Analysis

A machine learning project that combines classical classification and biomedical NLP to predict lung cancer risk and extract medical entities from clinical text.


## Overview

This project covers two parallel approaches:

1. **Classical ML Pipeline** — Generates a synthetic lung cancer dataset and trains a Logistic Regression model to predict cancer presence based on patient symptoms and lifestyle factors.
2. **Biomedical NLP Pipeline** — Uses pre-trained Hugging Face Transformer models for Named Entity Recognition (NER) on clinical text and fine-tunes BERT for healthcare sentence classification.


## Features

- Synthetic dataset generation with 300 patient records and 14 features
- Logistic Regression model with accuracy, confusion matrix, and classification report
- Feature importance visualization (model coefficients)
- Actual vs. predicted distribution comparison
- Biomedical NER using `d4data/biomedical-ner-all` to detect diseases, drugs, and symptoms
- BERT fine-tuning (`bert-base-uncased`) for disease vs. drug sentence classification
- ROC curve and F1 evaluation metrics


## Dataset

The synthetic dataset includes the following features:

| Feature | Description |
|---|---|
| AGE | Patient age (30–79) |
| SMOKING | Smoking status (0/1) |
| YELLOW_FINGERS | Yellow fingers indicator |
| ANXIETY, PEER_PRESSURE | Behavioral factors |
| CHRONIC_DISEASE | Pre-existing conditions |
| FATIGUE, ALLERGY, WHEEZING | Physical symptoms |
| ALCOHOL_CONSUMING | Lifestyle factor |
| COUGHING, SHORTNESS_OF_BREATH | Respiratory symptoms |
| SWALLOWING_DIFFICULTY, CHEST_PAIN | Advanced symptoms |
| LUNG_CANCER | Target variable (0 = No, 1 = Yes) |

> Target is derived from a rule: if `SMOKING + CHRONIC_DISEASE + CHEST_PAIN + COUGHING > 2`, then `LUNG_CANCER = 1`.


## Tech Stack

- Python 3.x
- `pandas`, `numpy` — data handling
- `scikit-learn` — model training and evaluation
- `matplotlib`, `seaborn` — visualization
- `transformers`, `datasets`, `evaluate` — Hugging Face NLP pipeline



## Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn transformers datasets evaluate
```


## Usage

Run the notebook cells in order:

1. **Generate dataset** → `lung_cancer_generated.csv`
2. **Train Logistic Regression** → view accuracy, confusion matrix, feature importance
3. **Run Biomedical NER** → extract entities from clinical text
4. **Fine-tune BERT** → classify healthcare sentences as disease or drug-related
5. **Evaluate** → accuracy, F1 score, confusion matrix

## Results

- Logistic Regression trained on 80/20 train-test split
- BERT fine-tuned for 2 epochs on a small healthcare sentence dataset
- Evaluation metrics: Accuracy, F1 Score, Confusion Matrix


## Project Structure

```
lung_cancer.ipynb        # Main notebook
lung_cancer_generated.csv  # Auto-generated synthetic dataset (created on run)
```


## Disclaimer

This project uses **synthetic data** for educational purposes only. It is not intended for real medical diagnosis or clinical decision-making.
