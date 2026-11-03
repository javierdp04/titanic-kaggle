import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
dataDirectory = Path.cwd() / 'data'

trainData = pd.read_csv(dataDirectory / 'train.csv').copy()

trainData.dropna(subset=['Embarked'])

X = trainData.drop(columns=['Fare', 'Sex', 'Pclass'])
y = trainData['Embarked']

num_cols = ['Fare']
cat_cols = ['Pclass', 'Sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

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
        ("numerical", numerical_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols)
])

main_pipeline = Pipeline([
    ("preprocessing", preprocessing_transformer),
    ("classifier", KNeighborsClassifier())
])