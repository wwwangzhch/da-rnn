import numpy as np


def cal_RMSE(pred, target):
    return np.sqrt(np.mean((np.array(pred)-np.array(target))**2))


def cal_NRMSE(pred, target):
    a = np.sqrt(np.mean((np.array(pred)-np.array(target))**2))
    b = np.mean(np.abs(target))
    return a/b


def cal_MAPE(pred, target):
    diff = np.abs(np.array(target)-np.array(pred))
    return np.mean(diff / np.abs(target))


def cal_SMAPE(pred, target):
    diff = np.abs(np.array(target)-np.array(pred))
    absolute = (np.array(target) + np.array(pred))/2
    result = np.mean(diff / absolute)
    return result


def calculate_eval(pred, target):
    RMSE = cal_RMSE(pred, target)
    NRMSE = cal_NRMSE(pred, target)
    MAPE = cal_MAPE(pred, target)
    SMAPE = cal_SMAPE(pred, target)
    return RMSE, NRMSE, MAPE, SMAPE
