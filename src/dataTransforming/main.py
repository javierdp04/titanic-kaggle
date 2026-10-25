import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

columnsToEncode = ['Sex', 'Embarked']
columnsToZScale= ['Age', 'Fare']
columnsToLogScale = ['Fare']

def transformData(trainData, testData):

    encodedTrainData, encodedTestData = oneHotEncode(trainData, testData, columnsToEncode)

    scaledTrainData = logScale(encodedTrainData, columnsToLogScale)
    scaledTestData = logScale(encodedTestData, columnsToLogScale)

    scaledTrainData[columnsToZScale], scaledTestData[columnsToZScale] = zScale(scaledTrainData, scaledTestData, columnsToZScale)

    return scaledTrainData, scaledTestData


def oneHotEncode(trainData, testData, columnsToEncode):
    encoder = OneHotEncoder(sparse_output=False)    
    encoder.fit(trainData[columnsToEncode])

    return _applyEncoder(trainData, encoder, columnsToEncode), _applyEncoder(testData, encoder, columnsToEncode)

def _applyEncoder(data, encoder, columnsToEncode):
    df_cod = pd.DataFrame(
            encoder.transform(data[columnsToEncode]),
            columns=encoder.get_feature_names_out(columnsToEncode),
            index=data.index,          
        )
    return pd.concat([data.drop(columns=columnsToEncode), df_cod], axis=1)

def zScale(trainData, testData, columnsToScale):
    scaler = StandardScaler().fit(trainData[columnsToScale])
    return scaler.transform(trainData[columnsToScale]), scaler.transform(testData[columnsToScale])

def logScale(data, columnsToScale):
    data = data.copy()
    for column in columnsToScale:
        data[column] = np.log1p(data[column])
    return data









    
    


