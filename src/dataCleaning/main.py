import numpy as np
import pandas as pd

train_data = pd.read_csv('../../data/train.csv')
test_data = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')



train_data['Age'].fillna(train_data['Age'].mean())