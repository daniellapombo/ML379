# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:57:28 2019

@author: danie
"""

from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

print(X_train.shape)
print(y_train.shape)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
print("X_Train", X_train.shape, X_train.size)
print()
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

