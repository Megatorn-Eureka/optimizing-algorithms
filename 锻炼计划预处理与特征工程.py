import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif

# 导入表格
dailyActivity_merged = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_merged.csv")

# 获取列名（获得所有的特征名称）
columns_name = dailyActivity_merged.columns.values

"预处理，采用StandardScaler"

# 数据标准化
scaler = StandardScaler()                              #实例化
dailyActivity_scaler = scaler.fit_transform(dailyActivity_merged)              #使用fit_transform(data)一步达成结果

"特征选择"

# 删除方差为0的数据
selector = VarianceThreshold()                  #实例化
dailyActivity_selector = selector.fit_transform(dailyActivity_scaler)     #删除不合格特征

# 保留特征名称
select_name_index0 = selector.get_support(indices=True)  # 留下特征的索引值，list格式
select_name0 = []
for i in select_name_index0:
    select_name0.append(columns_name[i])

# 恢复列名（特征名称）
dailyActivity_selector = pd.DataFrame(data=dailyActivity_selector ,columns=select_name0)       #指定行名（index）和列名（columns）
# print(dailyActivity_selector)

# 导出数据
dailyActivity_selector = pd.DataFrame(dailyActivity_selector)
dailyActivity_selector.to_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_selector.csv")
