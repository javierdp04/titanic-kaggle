import numpy as np
import pandas as pd

def cleanData(trainData : pd.DataFrame):
    trainData = trainData.copy()
    trainData['Age'] = trainData['Age'].fillna(trainData['Age'].median())
    trainData['Embarked'] = trainData['Embarked'].fillna('S')
    return trainData