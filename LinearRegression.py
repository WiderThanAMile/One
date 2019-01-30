# from https://towardsdatascience.com/linear-regression-the-easier-way-6f941aa471ea
# also https://github.com/Sagarsharma4244/Linear-Regression-Sklearn-Python


import pandas as pd
import pprint
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split

boston = load_boston()
pprint.pprint(boston)
df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

model = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4 )

model.fit(x_train,y_train)

results = model.predict(x_test)
#print(results[4])
print("result=" , results)
print("y_test=" , y_test)