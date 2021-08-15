import numpy as np
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    """
    二つのベクトルを比較してRMSEを計算
    :return: RMSE
    """
    return np.sqrt(mean_squared_error(y_test, y_pred))

