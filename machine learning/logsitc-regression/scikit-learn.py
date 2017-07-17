from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
print(iris.keys())
x = iris["data"][:,3:]
y = (iris["target"]==2).astype(np.int)
log_reg = LogisticRegression()
log_reg.fit(x,y)
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_log_proba(x_new)
plt.plot(x_new,y_proba[:,1],"g-", label ="Iris-Virginica")
plt.plot(x_new,y_proba[:,0],"b--", label = "Not Iris-Virginica")
prediction = log_reg.predict([[0.5]])
print(prediction)
plt.show()
