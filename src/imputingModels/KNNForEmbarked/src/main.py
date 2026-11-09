import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from pathlib import Path

NUM_COLS = ['Fare']
CAT_COLS = ['Pclass', 'Sex']

def build_embarked_model(n_neighbors=5, weights='uniform', p=2):
    numerical_pipeline = Pipeline([
        ("numerical_simple_imputer", SimpleImputer(strategy='median')),
        ("numerical_log_transformer",FunctionTransformer(np.log1p)),
        ("numerical_standar_scaler",StandardScaler())
    ])
    categorical_pipeline = Pipeline([
    ("categorical_simple_imputer", SimpleImputer(strategy='most_frequent')),
        ("categorical_onehot_encoder", OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessing_transformer = ColumnTransformer([
        ("numerical", numerical_pipeline, NUM_COLS),
        ("categorical", categorical_pipeline, CAT_COLS)
    ])

    main_pipeline = Pipeline([
        ("preprocessing", preprocessing_transformer),
        ("classifier",KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p))
    ])
    return main_pipeline

class KNNEmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self,feature_cols=['Fare', 'Pclass', 'Sex']):
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        self.model = build_embarked_model(n_neighbors=17, weights='distance', p=1)
        y = X['Embarked'].notna()
        self.model.fit(X.loc[y, self.feature_cols], X.loc[y, 'Embarked'])
        return self

    def transform(self, X):
        X = X.copy()
        y = X['Embarked'].isna()
        if y.any():
            X.loc[y, 'Embarked'] = self.model.predict(X.loc[y, list(self.feature_cols)])
        return X[['Embarked']]


if __name__ == '__main__':
    dataDirectory = Path.cwd() / 'data'
    
    trainData = pd.read_csv(dataDirectory / 'train.csv').copy()

    trainData = trainData.dropna(subset=['Embarked'])
    
    X = trainData[NUM_COLS + CAT_COLS]
    y = trainData['Embarked']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

    param_grid = [
        {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        }
    ]
    grid_search = GridSearchCV(build_embarked_model(), param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)  
    
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor score CV:", grid_search.best_score_)
    y_predict = grid_search.predict(X_test)
    print("Resultado en el test", accuracy_score(y_test, y_predict))

