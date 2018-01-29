from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


cancer=load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.7)


clf=tree.DecisionTreeClassifier()

plt.figure(figsize=(10,5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='viridis')
plt.show()

clf=clf.fit(cancer.data, cancer.target)

