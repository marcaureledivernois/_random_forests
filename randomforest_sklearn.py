from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import sys
import os
import sklearn
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
'''
Decision tree with sklearn
'''

#---------------------- Load Dataset
iris = datasets.load_iris()
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True) # remove any empty lines
iris_X=iris_df.iloc[:,[0,1,2,3]]

#---------------------- Format
# Format y,X as ndarray of shape (number of samples, number of features)
y = iris_df['class'].values
y = np.reshape(y, (-1,1))
X = iris_df.iloc[:,0:4].values

print('Shape X : ', X.shape, '\nShape y : ', y.shape)

#----------------------- Train/test split
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, stratify=y, random_state=seed) #stratify respects y proportions

