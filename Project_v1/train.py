import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn import datasets

import warnings
warnings.filterwarnings(action="ignore")

dataset = datasets.load_iris()
X, y = dataset.data, dataset.target
print("X shape: ", X.shape)
print("y shape: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train shape : ", X_train.shape)
print("y_train shape : ", y_train.shape)
print("X_test shape : ", X_test.shape)
print("y_test shape : ", y_test.shape)

clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Model validation Score : ", score)

# 保存模型
joblib.dump(clf, "linear_regression_model.pkl")
