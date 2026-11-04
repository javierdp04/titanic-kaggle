from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from imputingModels.KNNForEmbarked.src.main import KNNEmbarkedImputer
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


FEATURES_TO_DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin']
NUMERICAL_COLS = ['Age']
NUMERICAL_COLS_LOG_SCALE = ['Fare']
CATEGORICAL_COLS = ['Sex', 'SibSp', 'Pclass', 'Parch']

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def build_main_model():
    numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
        ("normalization", StandardScaler())
    ])
    numerical_pipeline_with_log_scaling = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaling", FunctionTransformer(np.log1p)),
        ("normalization", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("ohe", OneHotEncoder(handle_unknown='ignore'))
    ])

    embarked_pipeline = Pipeline([
        ("imputer", KNNEmbarkedImputer()),
        ("ohe", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessing_transformer = ColumnTransformer([
        ("drop", "drop", FEATURES_TO_DROP),
        ("numerical", numerical_pipeline, NUMERICAL_COLS),
        ("numerical_pipeline_with_log_scaling", numerical_pipeline_with_log_scaling, NUMERICAL_COLS_LOG_SCALE),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ("embarked_feature", embarked_pipeline, [ 'Sex', 'Fare', 'Pclass', 'Embarked'])
    ])

    main_pipeline = Pipeline([
        ("preprocessing", preprocessing_transformer),
        ("classifier", RandomForestClassifier(random_state=0))
    ])
    return main_pipeline   

if __name__ == '__main__':
    trainData = pd.read_csv(DATA_DIR / 'train.csv')
    
    X = trainData.drop(columns=['Survived']).copy()
    y = trainData['Survived'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    main_pipeline = build_main_model()
    param_grid = [
        {
            "classifier__n_estimators": [500, 700, 1000],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", 0.5, 0.75],
        }
    ]
    grid_search = GridSearchCV(main_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    n_fits = len(ParameterGrid(param_grid)) * 10
    with tqdm_joblib(tqdm(total=n_fits, desc="Ajustando hiperparametros")):
        grid_search.fit(X_train, y_train)

    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor score CV:", grid_search.best_score_)
    y_predict = grid_search.predict(X_test)
    print("Resultado en el test", accuracy_score(y_test, y_predict))


