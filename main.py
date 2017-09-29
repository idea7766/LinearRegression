from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

import linear_model

# 讀寫 boston 資料，data 和 target 皆為 np array
boston = datasets.load_boston()
x = boston.data
y = boston.target
print('X :', x)
print('Y :', y)

b, w = linear_model.LinearRegression(x, y, lr = 0.00001, epoch = 5000)
print('b', b)
print('w', w)

y_pred = linear_model.predcit(x, b, w)

mse = linear_model.mse(y, y_pred)
print(mse)

fig, ax = plt.subplots()
ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()