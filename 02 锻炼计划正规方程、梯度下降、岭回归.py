from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pandas import Series, DataFrame
import joblib
import pandas as pd
import numpy as np

def linear1():
    # 正规方程的优化方法
    # 获取数据
    dailyActivity_selector = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_selector.csv")
    Calories = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\Calories.csv")

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(dailyActivity_selector, Calories, random_state=22)

    # 预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 得出模型
    print("正规方程权重系数为：\n", estimator.coef_)
    print("正规方程偏置为： \n", estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测卡路里：\n", y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("正规方程均方误差为：\n",error)

def linear2():
    # 梯度下降的优化方法对波士顿放假进行预测
    # 获取数据
    dailyActivity_selector = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_selector.csv")
    Calories = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\Calories.csv")

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(dailyActivity_selector, Calories, random_state=22)

    # 预估器
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 得出模型
    print("梯度下降权重系数为：\n", estimator.coef_)
    print("梯度下降偏置为： \n", estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测卡路里：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降均方误差为：\n", error)

def linear3():
    # 岭回归的优化方法对波士顿放假进行预测
    # 获取数据
    dailyActivity_selector = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\dailyActivity_selector.csv")
    Calories = pd.read_csv(r"C:\Users\Lenovo\Desktop\FitBit Fitness Tracker Data(FitBit Fitness追踪器数据）\Calories.csv")

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(dailyActivity_selector, Calories, random_state=22)

    # 预估器
    estimator = Ridge()
    estimator.fit(x_train, y_train)

    # 保存模型
    joblib.dump(estimator, "my_ridge.pkl")
    # 加载模型
    estimator = joblib.load("my_ridge.pkl")

    # 得出模型
    print("岭回归权重系数为：\n", estimator.coef_)
    print("岭回归偏置为： \n", estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测卡路里：\n", y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("岭回归均方误差为：\n",error)


if __name__ == "__main__":

    # 正规方程
    linear1()

    # 梯度下降
    linear2()

    # 岭回归
    linear3()
