from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from pandas import Series, DataFrame
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
import pandas as pd
import numpy as np

# 获取数据（利用pd读取数据）
dailyActivity_merged = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_merged.csv")
sleepDay_merged = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\sleepDay_merged.csv")

# 使用merge函数将几张表合并
# on为共同的列名
# how为外连接，没有值则用NaN填充
# sort=False为不需要排序
dailyActivity_sleepDay_merged = pd.merge(dailyActivity_merged, sleepDay_merged, how='outer', on='Id_and_Date', sort=False)
dailyActivity_sleepDay_merged.to_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_sleepDay_merged.csv")