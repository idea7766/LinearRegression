from sklearn import datasets
import numpy as np

import model

# 讀寫 boston 資料，data 和 target 皆為 np array
boston = datasets.load_boston()
x = boston.data
y = boston.target

model.LinearRegression()