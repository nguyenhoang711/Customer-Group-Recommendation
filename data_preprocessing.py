import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

def check_and_drop_duplicated(df):
    print("Số lượng dữ liệu trùng lặp: ", df.duplicated().sum())
    df = df.drop_duplicates()
    return df


def check_null(df):
    print("Số lượng dữ liệu trống:\n", df.isna().sum())


def remove_unnecess_cols(df, cols):
    df = df.drop(columns=cols, axis=1)
    return df

def apply_log(data):
    return np.log(data)

def normality_test(column):
    return stats.normaltest(column)

def power_transform(column):
    feature = column.to_numpy().reshape(-1,1)
    powtr = PowerTransformer()
    feature_transf = powtr.fit_transform(feature)
    array_1d = feature_transf.flatten()
    feature = pd.Series(data=array_1d, index=list(range(len(array_1d))))
    return feature


def min_max_scaling(data):
    scaler = MinMaxScaler()
    data_normed = scaler.fit_transform(data)
    return data_normed

def z_score_standard(data):
    scaler = StandardScaler()
    data_standardlized = scaler.fit_transform(data)
    return data_standardlized

