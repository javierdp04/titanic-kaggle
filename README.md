# Titanic – Kaggle

Survival classifier for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition, using a `RandomForestClassifier` inside a scikit-learn pipeline.

## Structure

```
data/                     # train.csv, test.csv (not versioned)
src/
  main.py                 # Main pipeline (preprocessing + Random Forest) and evaluation
  helpers.py              # Statistical tests for the EDA (Chi², ANOVA, outliers)
  validation.ipynb        # Confusion matrix and cross-validated metrics
  dataExploration/EDA.ipynb
  imputingModels/KNNForEmbarked/
    src/main.py           # KNN imputer for 'Embarked' and its hyperparameter tuning
submissions/
  main.py                 # Trains on the full train.csv and generates the submission
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from Kaggle and place them in `data/`.

## Usage

Run every script **from the repo root** as a module (`-m`) so the `src` imports resolve:

```bash
python -m src.main                                   # Train and evaluate on a 20% hold-out
python -m src.imputingModels.KNNForEmbarked.src.main # Tune the KNN that imputes 'Embarked'
python -m submissions.main                           # Generate submission.csv for Kaggle
```

## Model

**Preprocessing** (`ColumnTransformer`):

| Columns | Treatment |
|---|---|
| `PassengerId`, `Name`, `Ticket`, `Cabin` | Dropped |
| `Age` | Median imputation + `StandardScaler` |
| `Fare` | Median imputation + `log1p` + `StandardScaler` |
| `Sex`, `SibSp`, `Pclass`, `Parch` | Most-frequent imputation + One-Hot |
| `Embarked` | KNN imputation (from `Fare`, `Pclass`, `Sex`) + One-Hot |

**Classifier:** `RandomForestClassifier(n_estimators=500, max_depth=7, max_features=0.75, min_samples_leaf=1)`. Hyperparameters were chosen with `GridSearchCV` (5 folds); the search code is commented out in `src/main.py`.

**Result:** ~0.83 accuracy on the hold-out set (20% of `train.csv`, `random_state=0`).

All preprocessing lives inside the pipeline, so during cross-validation it is fitted only on each fold's training data, with no data leakage.
