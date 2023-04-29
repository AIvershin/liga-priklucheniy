import numpy as np 
import pandas as pd

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from model_functions import *
from model_features import *

def median_absolute_error(y_true,y_pred):
    """
    MedianAE
    """
    
    return np.median(abs(np.array(y_true)-np.array(y_pred)))

def median_absolute_percentage_error(y_true,y_pred):
    """
    MedianAPE
    """
    ape=abs(np.array(y_true)-np.array(y_pred))/np.array(y_true)
    
    return np.median(ape)

def r2_adjusted(y,X,model):
    """
    Скорректированный R2
    """
    # Прогноз
    y_pred=model.predict(X)
    # Параметры для r2_adj
    r2=r2_score(y,y_pred)
    n=len(X)
    p=model.n_features_in_
    
    return 1-(1-r2)*(n-1)/(n-p)