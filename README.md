## Random Forests

# Overview

Supervised. Random forest is a set of decision trees. 

Decision tree discriminates a sample by asking binary questions (e.g : feature1 > threshold ?).
Optimal "questions" are obtained by minimizing Gini index. 

One problem that might occur with one big (deep) single Decision Tree is that it can overfit. 
That is the DT can “memorize” the training set. The point of Random Forest is to prevent overfitting. 
It does this by creating random subsets of the features and building smaller
trees using the subsets and then it combines the subtrees (ensemble method).

# Gini index

If a data set D contains samples from C classes, gini index is defined as:
gini(D) = 1 - &Sigma;<sub>c=1</sub> <sup>C</sup> 𝑷<sup>2</sup>
where P is the relative frequency of class c in D.

If a data set D splits on S into two subsets D1 and D2, the gini index is defined as:
gini(D) = D1/D * gini(D1) + D2/D * gini(D2)

# Bootstrap 

The trees are grown from different subsets of the training data by a bootstrap aggregating (i.e "bagging") procedure
(Breiman, 1996) which ensures the diversity of the trees and minimizes the similarities between
them. In practice, we randomly select (with replacement = a datapoint can be drawn multiple times) 
subsets from the training set, then train a bunch of decision trees.
The final prediction is an average (or majority vote) of the predictions of every decision tree (= ensemble method)


## Credits

* Siraj Raval
* [rushter](https://github.com/rushter)
* https://www.es.utoronto.ca/wp-content/uploads/2019/03/Lecture_09.pdf
