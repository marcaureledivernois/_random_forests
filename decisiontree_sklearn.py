from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import sys
import os.path
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
import matplotlib.pyplot as plt
import streamlit as st

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

#----------------------- Grid Search

paramgrid = dict(min_samples_leaf = [1,3],
                 min_samples_split=[2,5],
                 max_depth = [1,20],
                 max_features = [1,4])
clf = DecisionTreeClassifier(criterion='gini')
grid = GridSearchCV(estimator=clf,param_grid=paramgrid,scoring='accuracy',cv=5)
grid.fit(X_train,y_train)

print('=================RESULTS=============')
print("grid.cv_results_ {}".format(grid.cv_results_))
print('===============BEST PARAMS===================')
print('The parameters combination that would give best accuracy is : ')
print(grid.best_params_)
print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)

#----------------------- Optimal tree

optimal_tree = DecisionTreeClassifier(criterion='gini',
                                      max_depth=20,
                                      max_features=4,
                                      min_samples_leaf=1,
                                      min_samples_split=2)

optimal_tree.fit(X_train,y_train)

#----------------------- Out of sample performance

#accuracy
y_pred = optimal_tree.predict(X_test)
print('======== ACCURACY ===========')
print(accuracy_score(y_test,y_pred))
print('======= CONFUSION MATRIX ====')
print(confusion_matrix(y_test,y_pred))

# confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()

# precision recall
probas = optimal_tree.predict_proba(X_test)
skplt.metrics.plot_precision_recall(y_test, probas)
plt.show()

#----------------------- Feature importance

print('Feature importances : ' , optimal_tree.feature_importances_)

skplt.estimators.plot_feature_importances(optimal_tree, feature_names=['petal length', 'petal width','sepal length', 'sepal width'])
plt.show()

#----------------------- Plot Optimal Tree

st.write(confusion_matrix(y_test,y_pred))
