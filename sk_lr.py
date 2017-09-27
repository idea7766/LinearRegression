from sklearn import datasets
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# 讀寫 boston 資料，data 和 target 皆為 np array
boston = datasets.load_boston()
x = boston.data
y = boston.target
print('X :', x)
print('Y :', y)

lr = linear_model.LinearRegression()

lr.fit(x, y)
predicted = lr.predict(x)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()