# https://github.com/Sagarsharma4244/Cross-Validation/blob/master/Visualization_cross_validation.py
# https://towardsdatascience.com/cross-validation-code-visualization-kind-of-fun-b9741baea1f8

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict

model = linear_model.LinearRegression()

boston = datasets.load_boston()
print(boston)
y = boston.target

# cross_val_predict with 10-fold cross validation (cv)
predicted = cross_val_predict(model, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
