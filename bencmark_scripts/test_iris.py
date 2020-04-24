import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import PackedForest
import sys
import compiledtrees

"""
iris = load_iris()
X = iris.data
X = np.asarray(X, dtype=np.float32)
Y = iris.target
print("X Shape", X.shape)
print("Y Shape", Y.shape)
"""

mnist = fetch_openml('mnist_784')
X = mnist.data[:1000]
Y = mnist.target[:1000]

clf = RandomForestClassifier(n_estimators=8)
# clf = RandomForestClassifier(n_estimators = 3, max_depth = 2, random_state = 19832312)
# clf = RandomForestClassifier(n_estimators = 2, max_depth = 2, random_state = int(sys.argv[1]))

clf.fit(X,Y)
#for i in range(0,len(clf.estimators_)):
#    clf.estimators_[i].tree_.predict(np.asarray(X, dtype=np.float32))

frst = PackedForest(forest_classifier=clf, interleave_depth=2, n_bins=2)

# compiled_predictor = compiledtrees.CompiledRegressionPredictor(clf)
# predictions = compiled_predictor.predict(X)

print(clf.n_estimators)
print("Prediction is")
#a = clf.predict(X[60:63,:])
#b = frst.predict(X[60:63,:])

a = clf.predict(X[5:6])
b = frst.predict(X[5:6])

print (a)
print (b)


for i in range(4):
    a = clf.predict(X[i:i+1])
    b = frst.predict(X[i:i+1])
    b = np.asarray(b, np.intp)
    b = list(map(str, np.asarray(b, dtype=np.int)))
    # print(a, b)
    if a[0]!=b[0]:
        print("Wrong")
"""
print(a)
print(np.asarray(b, np.int))
#print(np.equal(a, b))
#print(np.sum(np.equal(a, b)))

#print(clf)

print("orig vs a")
print(np.sum(np.equal(Y, a)))
print("orig vs b")
print(np.sum(np.equal(Y, b)))
"""
