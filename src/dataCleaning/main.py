import numpy as np
import pandas as pd

def clean_data(trainData : pd.DataFrame):
    trainData['Age'] = trainData['Age'].fillna(trainData['Age'].median())
    trainData['Embarked'] = trainData['Embarked'].fillna('S')
    return trainData.drop(labels=['Cabin'])