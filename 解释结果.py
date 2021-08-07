import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn import preprocessing

# 导入表格
dailyActivity_merged = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_merged.csv")

# 获取列名（获得所有的特征名称）
columns_name = dailyActivity_merged.columns.values

result_data = np.array([-1.50221155, 4.42117265, -0.64266334, 3.61310579, 5.75237521, 6.47928116, -1.76703401, 0.61860734])

# 计算原始数据每行和每列的均值和方差，data是多维数据
scaler = preprocessing.StandardScaler().fit(dailyActivity_merged)

# 得到每列的平均值,是一维数组
mean = scaler.mean_
mean = np.array(mean)
# print("均值为：", mean)

# 得到每列的标准差,是一维数组
std = scaler.var_
std = np.array(std)
# print("标准差为：", std)

# 得出结果
origin_data = result_data*std + mean
print(origin_data)
# 输出：
# [-3.88268917e+07  3.27269778e+01  6.63399307e-02  1.83707768e+01
#   6.22011683e+03  2.59926018e+03 -2.08462631e+04  5.70775625e+04]
