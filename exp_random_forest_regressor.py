#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:54:05 2017

@author: hangpinglin
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

data = np.genfromtxt('/Users/hangpinglin/big/LargeTrain.csv',delimiter=',')
col_row = data.shape
row = col_row[0]
col = col_row[1]
print (row,col)
X= data[1:row,0:col-1]
Y= data[1:row,col-1:col]
y=Y.ravel()

# Build a forest and compute the feature importances
forest = RandomForestRegressor()
forest.fit(X, y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
cut_indices = indices[0:10]

# Print the feature ranking
print("Feature ranking:")

for f in range(30):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[cut_indices],
       color="r", yerr=std[cut_indices], align="center")
plt.xticks(range(10), cut_indices)
plt.xlim([-1, 10])
plt.show()