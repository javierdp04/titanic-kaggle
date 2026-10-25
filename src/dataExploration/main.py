import numpy as np
import pandas as pd
from  pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns   

dataDirectoty = Path(__file__).resolve().parents[2] / 'data'

trainData = pd.read_csv(dataDirectoty / 'train.csv')
testData = pd.read_csv(dataDirectoty / 'test.csv')

# print(trainData.duplicated().sum())

# print(trainData.head(10))
# print(trainData.info()) 
#print(trainData.describe())

#sns.histplot(data=testData, x="Age")
#plt.show()

#print(trainData['Survived'].corr(trainData['Age']))

# trainDataWithFloorColumn = trainData
# trainDataWithFloorColumn['Floor'] = trainData['Cabin'].str[0]
# sns.catplot(data=trainDataWithFloorColumn, kind='count', x='Floor', hue='Survived')
# plt.show()

# sns.countplot(data=trainData, x="Embarked",stat='percent')
# plt.show()


# --- Categorical Features ---

# sns.countplot(data=trainData, x='Sex', hue='Survived')
# plt.show()

#sns.countplot(data=trainData, x='Pclass', hue='Survived')
#plt.show()

#sns.countplot(data=trainData, x='Embarked', hue='Survived')
#plt.show()