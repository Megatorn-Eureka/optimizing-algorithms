import pandas as pd

data = pd.read_csv(r'C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_BMI_merged.csv')
data_not_nan = data[data['BMI'].notna()]
data_not_nan.to_csv(r'C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_BMI_merged.csv')
