import numpy as np
import pandas as pd
from  pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns   
from scipy.stats import chi2_contingency
from scipy .stats import f_oneway

def chi2_test(data, var1, var2, alpha):
    data = data[[var1, var2]].copy().dropna()
    chi2, p_value, dof, exp = chi2_contingency(pd.crosstab(data[var1], data[var2]))
    print("Chi² test:")
    print(f"\tH0: {var1} and {var2} are independent")
    print(f"\tH1: {var1} and {var2} are dependent")
    print(f"\tWe reject H0" if p_value <= alpha else "We do not reject H0")
    print(f"\tP value: {p_value}")


def anova_test(data, categoricaVar, numericalVar, alpha):
    data = data[[categoricaVar, numericalVar]].dropna()
    grupos = [g[numericalVar].values for _, g in data.groupby(categoricaVar)]
    F, p_value = f_oneway(*grupos)

    print("ANOVA test:")
    print(f"\tH0: {categoricaVar} and {numericalVar} are independent")
    print(f"\tH1: {categoricaVar} and {numericalVar} are dependent")
    print(f"\tWe reject H0" if p_value <= alpha else "We do not reject H0")
    print(f"\tF: {F}, p value: {p_value}")

def are_there_outliers(data, variable):
    q1, q3 = data[variable].quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((data[variable] < q1 - 1.5 * iqr) | (data[variable] > q3 + 1.5 * iqr)).any()