from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, PackedForest
import numpy as np
import sys
from sklearn.datasets import load_iris
from operator import add
from functools import reduce
from datetime import datetime
import treelite.gallery.sklearn
import treelite.runtime     # runtime module
import compiledtrees
from test_treelite import *
from sklearn.utils import shuffle
import random
import multiprocessing

custom_data_home = '.'
toolchain = 'gcc'
cache_size = (1024*1024)*8      # MB * 8
repetitions = 2


def flush_cache(arr):
    # print(arr.size)
    for i in range(0, arr.size):
        arr[i] = random.random()


def benchmark_cifar_small(n_estimators = 2048, interleave_depth = 2, batch_size = 1, n_threads = 8):
    print("Fetching cifar small")
    cifar = fetch_openml('cifar_10')
    print(cifar.data.shape)
    X = cifar.data
    Y = cifar.target

    X_train = X[:50000]
    X_test = X[50000:]

    Y_train = Y[:50000]
    Y_test = Y[50000:]

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=int(sys.argv[1]))
    print("Fitting")
    clf.fit(X_train, Y_train)

    # packed_forest(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    compiled_trees(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size,
                   n_threads=n_threads)



def benchmark_higgs(n_estimators = 2048, interleave_depth = 2, batch_size = 1, n_threads = 8):
    print("Fetching higgs")
    higgs = fetch_openml('higgs')
    print(higgs.data.shape)
    print(higgs.target.shape)
    X = higgs.data
    Y = higgs.target
    print(X[~np.isnan(X).any(axis=1)].shape)

    Y = Y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]

    X_train = X[:(X.shape[0]-50000)]
    X_test = X[(X.shape[0]-50000):]
    print(X_train.dtype)
    Y_train = Y[:(X.shape[0]-50000)]
    Y_test = Y[(X.shape[0]-50000):]
    print(Y_train.dtype)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    clf = RandomForestClassifier(n_estimators=n_estimators)
    print("Fitting")
    clf.fit(X_train, Y_train)

    # sklearn_naive(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    packed_forest(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    # tree_lite(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    # compiled_trees(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)


def benchmark_cifar10(n_estimators = 2048, interleave_depth = 2, batch_size = 1, n_threads = 8):
    print("Fetching cifar")
    # cifar = fetch_openml('stl-10')
    # print(cifar.data.shape)
    # print(cifar.target.shape)
    #
    # X = cifar.data[:30000, :10]
    # Y = cifar.target[:30000]
    X, Y = load_csv(dataset_path)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(type(X), type(Y))
    print(len(X[0]))
    print(len(X), len(Y))
    print(X.dtype, Y.dtype)
    print(X.shape, Y.shape)
    print(Y[0])

    X_train = X[:50000]
    X_test = X[50000:]

    Y_train = Y[:50000]
    Y_test = Y[50000:]

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    # X = np.asarray(higgs.data[:, :20], dtype=np.float32)
    # Y = higgs.target

    clf = RandomForestClassifier(n_estimators=n_estimators)
    print("Fitting")
    clf.fit(X_train, Y_train)
    print("Classifier order is", clf.estimators_[0].classes_)

    # sklearn_naive(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    packed_forest(clf=clf, X_test=X_test, Y_test=Y_test,  interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    tree_lite(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    compiled_trees(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)


def benchmark_mnist(n_estimators = 2048, interleave_depth = 2, batch_size = 1, n_threads = 8):
    mnist = fetch_openml('mnist_784')
    print(mnist.data.shape)

    # mnist = load_iris()
    X = mnist.data
    Y = mnist.target

    X_train = X[:60000]
    X_test = X[60000:]

    print(X.dtype)
    print(Y.dtype)
    print(type(Y))

    Y_train = Y[:60000]
    Y_test = Y[60000:]

    print(X.shape, Y.shape)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    print("Fitting")
    clf.fit(X_train, Y_train)
    print("Classifier order is", clf.estimators_[0].classes_)

    # sklearn_naive(clf=clf, X_test=X_test, Y_test=Y_test, interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    # packed_forest(clf=clf, X_test=X_test, Y_test=Y_test,  interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    tree_lite(clf=clf, X_test=X_test, Y_test=Y_test,  interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)

    # compiled_trees(clf=clf, X_test=X_test, Y_test=Y_test,  interleave_depth=interleave_depth, batch_size=batch_size, n_threads=n_threads)


def compiled_trees(clf, X_test, Y_test, interleave_depth=2, batch_size=1, n_threads=8):
    print("Calling compiled trees")
    # batches = [batch_size]
    # batches = [10000, 5000, 1000, 500]
    batches = [10000, 5000, 1000, 500, 100, 10, 1]
    arr = np.empty(shape=(int)(cache_size/4), dtype=np.float32)  # 4bytes for n.float32

    # Shuffle dataset
    X_test, Y_test = shuffle(X_test, Y_test)

    # Compiling time
    tstart = datetime.now()
    compiled_predictor = compiledtrees.CompiledRegressionPredictor(clf)
    delta = (datetime.now() - tstart).total_seconds()
    print("compiledTrees: Packing Time (sec) is ", delta)

    print("--------------------------")
    for batch_size in batches:
        print("Batch size ", batch_size)
        rep_lst = []
        for reps in range(repetitions):
            compiled_trees_lst = []

            runs = (int)(Y_test.shape[0]/batch_size)

            for i in range(0, runs):
                # Flush cache before each run
                flush_cache(arr)
                tstart = datetime.now()
                compiled_predictor.predict(X_test[i:(i+1)*batch_size])
                delta = (datetime.now() - tstart).total_seconds()*1000  # ms
                compiled_trees_lst.append(delta)

            op_time = reduce(add, compiled_trees_lst)/Y_test.shape[0]
            print("compiledTrees normalized operation (ms) is ", op_time)
            rep_lst.append(op_time)
        print("compiledTrees:Avg prediction time (ms) for {0} is {1}".format(batch_size, reduce(add, rep_lst) / repetitions))

    # Check correctness

    # a = clf.predict(X_test)
    # b = compiled_predictor.predict(X_test)
    #
    # b = np.asarray(b, np.int)
    # # b = list(map(str, b))
    # # print(a[:100])
    # # print(b[:100])
    # print(np.sum(np.equal(a, b)))
    # print("orig vs a")
    # print(np.sum(np.equal(Y_test, a)))
    # print("orig vs b")
    # print(np.sum(np.equal(Y_test, b)))

def sklearn_naive(clf, X_test, Y_test, interleave_depth=2, batch_size=1, n_threads=8):
    # batches = [10000, 5000, 1000, 500]
    batches = [10000, 5000, 1000, 500, 100, 10, 1]
    arr = np.empty(shape=(int)(cache_size/4), dtype=np.float32)  # 4bytes for n.float32

    # Shuffle dataset
    X_test, Y_test = shuffle(X_test, Y_test)

    print("--------------------------")
    for batch_size in batches:
        print("Batch size ", batch_size)
        rep_lst = []
        for reps in range(repetitions):
            sklearn_naive_lst = []
            # Calculate for Sklearn Naive
            # ------------------------------------------------
            runs = (int)(Y_test.shape[0]/batch_size)
            for i in range(0, runs):
                # Flush cache before each run
                flush_cache(arr)
                tstart = datetime.now()
                clf.predict(X_test[i:(i+1)*batch_size])
                delta = (datetime.now() - tstart).total_seconds()*1000  # ms
                sklearn_naive_lst.append(delta)

            op_time = reduce(add, sklearn_naive_lst)/Y_test.shape[0]
            print("Naive operation is", op_time)
            rep_lst.append(op_time)
        print("SK-Native:Avg prediction time for {0} is {1}".format(batch_size, reduce(add, rep_lst)/repetitions))
        # ---------------------------------------------


def packed_forest(clf, X_test, Y_test, interleave_depth=2, batch_size=1, n_threads=8):
    print("Calling PackedForest")
    # batches = [batch_size]
    # batches = [10000, 5000, 1000, 500]
    batches = [10000, 5000, 1000, 500, 100, 10, 1]
    arr = np.empty(shape=(int)(cache_size/4), dtype=np.float32)  # 4bytes for n.float32

    # Shuffle dataset
    X_test, Y_test = shuffle(X_test, Y_test)

    # Packing time
    tstart = datetime.now()
    frst = PackedForest(forest_classifier=clf, interleave_depth=interleave_depth, n_bins=8)
    delta = (datetime.now() - tstart).total_seconds()
    print("PkdForest: Packing Time (sec) is ", delta)

    print("--------------------------")
    for batch_size in batches:
        print("Batch size ", batch_size)
        rep_lst = []
        for reps in range(repetitions):
            packed_forest_lst = []

            runs = (int)(Y_test.shape[0] / batch_size)

            for i in range(0, runs):
                # Flush cache before each run
                flush_cache(arr)
                tstart = datetime.now()
                frst.predict(X_test[i:(i+1)*batch_size], n_threads=n_threads)
                delta = (datetime.now() - tstart).total_seconds()*1000  # ms
                packed_forest_lst.append(delta)

            op_time = reduce(add, packed_forest_lst)/Y_test.shape[0]
            print("PackedForest normalized operation (ms) is", op_time)
            rep_lst.append(op_time)
        print("PkdForest:Avg prediction time (ms) for {0} is {1}".format(batch_size, reduce(add, rep_lst) / repetitions))

    # # print("Predicting")
    # a = clf.predict(X_test)
    # print(a[:5])
    # b = frst.predict(X_test)
    # # b = list(map(str, np.asarray(frst.predict(X_test), dtype=np.int)))
    # print(b[:5])
    #
    # print("a vs b")
    # print(np.sum(np.equal(a, b)))
    # print("orig vs a")
    # print(np.sum(np.equal(Y_test, a)))
    # print("orig vs b")
    # print(np.sum(np.equal(Y_test, b)))
    # #
    # # print("Printing outputs")
    # # print(a)
    # # print(b)


def tree_lite(clf, X_test, Y_test, interleave_depth=2, batch_size=1, n_threads=8):
    print("Calling treelite")
    # batches = [batch_size]
    # batches = [10000, 5000, 1000, 500]
    batches = [10000, 5000, 1000, 500, 100, 10, 1]
    arr = np.empty(shape=(int)(cache_size/4), dtype=np.float32)  # 4bytes for n.float32

    # Shuffle dataset
    X_test, Y_test = shuffle(X_test, Y_test)


    # Packing time
    tstart = datetime.now()
    # OLD------------------------
    # treelite_model = treelite.gallery.sklearn.import_model(clf)
    # print("Exporitng model")
    # treelite_model.export_lib(toolchain=toolchain, libpath='./mymodel_cifar.so', params={'parallel_comp': 32}, verbose=True)
    # print("Calling predictor class")
    # predictor = treelite.runtime.Predictor('./mymodel_cifar.so', verbose=True)
    # NEW------------------------
    model = process_model(clf)
    model.export_lib(toolchain=toolchain, libpath='./cifarmodel.so',
                     params={# 'annotate_in': './annotation.json',
                             'parallel_comp': 32}, verbose=True)
    predictor = treelite.runtime.Predictor(libpath='cifarmodel.so', verbose=True)
    # NEW------------------------
    delta = (datetime.now() - tstart).total_seconds()
    print("Treelite: Packing Time (sec) is ", delta)

    print("--------------------------")

    for batch_size in batches:
        print("Batch size ", batch_size)
        rep_lst =[]
        for reps in range(repetitions):
            treelite_lst = []

            runs = (int)(Y_test.shape[0]/batch_size)

            for i in range(0, runs):
                # Flush cache before each run
                flush_cache(arr)
                batch = treelite.runtime.Batch.from_npy2d(X_test, rbegin=i, rend=(i+1)*batch_size)
                tstart = datetime.now()
                predictor.predict(batch)
                delta = (datetime.now() - tstart)
                treelite_lst.append(delta)

            op_time = reduce(add, treelite_lst) / Y_test.shape[0]
            print("Treelite operation is", op_time)
            rep_lst.append(op_time)
        print("Treelite:Avg prediction time (ms) for {0} is {1}".format(batch_size, reduce(add, rep_lst) / repetitions))



    # out_pred = predictor.predict(batch)
    #
    # print("Shape of predictor is ", out_pred.shape)
    # print(out_pred[:5])
    #
    # out_pred = np.argmax(out_pred, axis=1)
    # out_pred = list(map(str, out_pred))
    # a = clf.predict(X_test)
    # print(a[:5])
    # print("Shape of native is", a.shape)
    # print(np.sum(np.equal(a, out_pred)))
    # print(out_pred[:5])
    # print("orig vs a")
    # print(np.sum(np.equal(Y_test, a)))
    # print("orig vs b")
    # print(np.sum(np.equal(Y_test, out_pred)))

    # -------------------------------------------------------


if __name__ == "__main__":
    # benchmark_mnist(n_estimators=16, interleave_depth=2, batch_size=10000, n_threads=multiprocessing.cpu_count())

    benchmark_cifar10(n_estimators=128, interleave_depth=2, batch_size=1, n_threads=multiprocessing.cpu_count())

    # benchmark_higgs(n_estimators=1, interleave_depth=2, batch_size=1, n_threads=multiprocessing.cpu_count())

    # benchmark_cifar_small(n_estimators=10, interleave_depth=2, batch_size=10000, n_threads=multiprocessing.cpu_count())
