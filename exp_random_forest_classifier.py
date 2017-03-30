#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:54:05 2017

@author: hangpinglin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

data = np.genfromtxt('/Users/hangpinglin/big/LargeTrain.csv',delimiter=',')
col_row = data.shape
row = col_row[0]
col = col_row[1]
print (row,col)
X= data[1:row,0:col-1]
Y= data[1:row,col-1:col]
y=Y.ravel()
y1=y.astype(int)
# Build a forest and compute the feature importances
forest = RandomForestClassifier()
forest.fit(X, y1)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
cut_indices = indices[0:10]

header_indices = np.copy(cut_indices)
top10_indices = np.append(header_indices,[-1])

df = pd.read_csv('/Users/hangpinglin/big/LargeTrain.csv')
out = df.iloc[:,top10_indices]
out.to_csv('/Users/hangpinglin/big/RF_classifier1.csv',index=False)

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[cut_indices],
       color="r", yerr=std[cut_indices], align="center")
plt.xticks(range(10), cut_indices)
plt.xlim([-1, 10])
plt.show()