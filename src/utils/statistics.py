import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fill_outliers(data, k, c):
  datanew = data.copy()
  Q1 = np.quantile(datanew.to_numpy().flatten(), .25)
  Q3 = np.quantile(datanew.to_numpy().flatten(), .75)
  IQR = Q3 - Q1
  lower = Q1 - c*IQR
  upper = Q3 + c*IQR

  for c in datanew.columns:
    # https://stackoverflow.com/questions/53806570/why-does-one-use-of-iloc-give-a-settingwithcopywarning-but-the-other-doesnt
    datanew.iloc[(datanew[c] < lower)|(datanew[c] > upper), datanew.columns.get_loc(c)] = np.nan

  print('how outliers ' + str(k) + ' = ' + str(datanew.isnull().sum().sum()))

  datanew = datanew.T.fillna(datanew.mean(axis=1)).T # Fill with the mean 
  return datanew

def describe_data(data):
    """
    Generate descriptive statistics for the given DataFrame
    """
    desc = pd.DataFrame()
    
    desc['Mean'] = data.mean(axis=1)
    desc['Median'] = data.median(axis=1)
    desc['Q1'] = data.quantile(0.25, axis=1)
    desc['Q3'] = data.quantile(0.75, axis=1)
    desc['Max'] = data.max(axis=1)
    desc['Min'] = data.min(axis=1)
    desc['Std'] = data.std(axis=1)
    desc['IQR'] = desc['Q3'] - desc['Q1']
    desc['Skewness'] = data.skew(axis=1)
    desc['Kurtosis'] = data.kurtosis(axis=1)
    desc['Range'] = desc['Max'] - desc['Min']   
    
    return desc