from dataCleaning.main import cleanData
from dataTransforming.main import transformData
from featureEngineering.main import dropFeatures
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

if __name__ == '__main__':
    trainData = pd.read_csv(DATA_DIR / 'train.csv')

    cleanedTrainData = cleanData(trainData)
    transformedTrainData = transformData(cleanedTrainData)
    transformedTrainData = dropFeatures(transformedTrainData)

    yTrain = transformedTrainData['Survived']
    XTrain = transformedTrainData.drop(columns=['Survived'])


    XTrain, XVal, yTrain, yVal = train_test_split(XTrain, yTrain, random_state=0)

    titanicModel = RandomForestClassifier(random_state=1)
    titanicModel.fit(XTrain, yTrain)

    print("Acurracy Score: ")
    print(accuracy_score(yVal, titanicModel.predict(XVal)))