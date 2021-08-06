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
from scipy import optimize as op

"利用线性规划求卡路里最大消耗量"

# 线性目标函数的系数:岭回归得出的均方误差最小的权重矩阵（向量）
weights = np.array([-0.52544171, 0.21088417, -0.25276103, 1.00602753, 0.58912286, 0.45856168, -0.29719819, 0.16309713])
# 不等式约束矩阵
distance_ub = np.array([[1,0,0,0,0,0,0,0], [0,1,1,1,0,0,0,0], [0,0,0,0,1,1,1,1]])
# 不等式约束向量
time_ub = np.array([5.581945412, 7.391615107, 11.08322969])
# 限定自变量范围
TotalSteps = (-1.50221155, 5.581945412)
LightActiveDistance = (-0.565443421, 7.682828205)
VeryActiveDistance = (-0.642663337, 6.695039234)
ModeratelyActiveDistance = (-1.638001996, 3.61310579)
VeryActiveMinutes = (-0.644733978, 5.752375208)
FairlyActiveMinutes = (-0.679033394, 6.479281157)
LightlyActiveMinutes = (-1.767034013, 2.980180801)
SedentaryMinutes = (-3.291886802, 1.490464004)

res = op.linprog(-weights, distance_ub, time_ub, bounds=(TotalSteps, LightActiveDistance, VeryActiveDistance, ModeratelyActiveDistance, VeryActiveMinutes, FairlyActiveMinutes, LightlyActiveMinutes, SedentaryMinutes))

print(res)
# 输出：重点关注的就是第一行和最后一行了，第一行是整个结果，最后一行是每个x的结果
# con: array([], dtype=float64)
# 为什么第一行是负的呢？原来这个函数其实是求最小值的，那么求最大值，怎么办呢？很简单，仔细观察的人应该发现，之前的函数里面，我写的是-weights，而不是weights。那么这个函数的出来的结果其实就是-weights的最小值，但很明显这恰恰是weights最大值的相反数。
# fun: -12.505062248961627
# message: 'Optimization terminated successfully.'
# nit: 6
# slack: array([7.08415696e+00, 8.89269103e-10, 7.06908310e-10])
# status: 0
# success: True
# x: array([-1.50221155, 4.42117265, -0.64266334, 3.61310579, 5.75237521,
#           6.47928116, -1.76703401, 0.61860734])
