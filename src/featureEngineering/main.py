import pandas as pd
import numpy as np

FEATURES_TO_DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin']

def dropFeatures(df):
    data = df.copy()
    return data.drop(columns=FEATURES_TO_DROP)