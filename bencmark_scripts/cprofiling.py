from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, PackedForest
import numpy as np
import sys
import random
import multiprocessing
import compiledtrees
from sklearn.utils import shuffle
from operator import add
from functools import reduce
from datetime import datetime
from bencmark_scripts.benchmark import flush_cache
from test_treelite import *
import cProfile, pstats
from io import StringIO

cache_size = (1024*1024)*8      # MB * 8
n_threads = multiprocessing.cpu_count()
interleave_depth = 2
batch_size = 10

X, Y = load_csv(dataset_path)
X = np.asarray(X)
Y = np.asarray(Y)

X_train = X[:50000]
X_test = X[50000:]

Y_train = Y[:50000]
Y_test = Y[50000:]

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Fit forest
clf = RandomForestClassifier(n_estimators=128)
print("Fitting")
clf.fit(X_train[:1000], Y_train[:1000])

#Create array for flushing cache
arr = np.empty(shape=(int)(cache_size/4), dtype=np.float32)  # 4bytes for n.float32


runs = (int)(Y_test.shape[0]/batch_size)
# runs = (int)(runs/10)
print("Packing")
frst = PackedForest(forest_classifier=clf, interleave_depth=interleave_depth, n_bins=n_threads)

# print("Flushing cache")
# tstart = datetime.now()
# flush_cache(arr)
# delta = (datetime.now() - tstart).total_seconds()
# print("Flushing cache time: (sec) is ", delta)

print("Enabling profiler")
pr = cProfile.Profile()
pr.enable()
for i in range(0, runs):
    flush_cache(arr)
    frst.predict(X_test[i:(i+1)*batch_size], n_threads=n_threads)
pr.disable()
s = StringIO()
# sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s)#.sort_stats(sortby)
ps.print_stats()
print(s.getvalue())


print("CT comilatoin")
compiled_predictor = compiledtrees.CompiledRegressionPredictor(clf)

# print("Flushing cache")
# tstart = datetime.now()
# flush_cache(arr)
# delta = (datetime.now() - tstart).total_seconds()
# print("Flushing cache time: (sec) is ", delta)

print("Enabling profiler")
pr = cProfile.Profile()
pr.enable()
for i in range(0, runs):
    flush_cache(arr)
    compiled_predictor.predict(X_test[i:(i+1)*batch_size])
pr.disable()
s = StringIO()
# sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s)#.sort_stats(sortby)
ps.print_stats()
print(s.getvalue())