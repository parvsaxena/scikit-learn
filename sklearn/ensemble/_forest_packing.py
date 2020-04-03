import numbers
from warnings import catch_warnings, simplefilter, warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel, delayed

from ..base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from ..metrics import r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                    ExtraTreeClassifier, ExtraTreeRegressor)
from ..tree._tree import DTYPE, DOUBLE
from ..utils import check_random_state, check_array, compute_sample_weight
from ..exceptions import DataConversionWarning
from ._base import BaseEnsemble, _partition_estimators
from ..utils.fixes import _joblib_parallel_args
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
from ._packed_forest import PkdForest
import multiprocessing

__all__ = ["PackedForest"]
n_bins = multiprocessing.cpu_count()
n_threads = multiprocessing.cpu_count()


class PackedForest:
    def __init__(self,
                 interleave_depth=1,
                 n_bins=n_bins,
                 forest_classifier=None):

        self.interleave_depth = interleave_depth
        self.n_bins = n_bins
        self.forest_classifier = forest_classifier
        # print(forest_classifier.estimators_)
        self.n_trees = forest_classifier.n_estimators
        self.tree_list = forest_classifier.estimators_
        self.classes_ = self.tree_list[0].classes_
        if self.n_trees < n_bins:
            self.n_bins = self.n_trees
        # for tree in self.tree_list:
        #     print(tree.tree_.node_count, tree.tree_.capacity)
        # print(self.tree_list)
        print("No of bins are", n_bins)

        self._pkd_forest = PkdForest([tree.tree_ for tree in self.tree_list],
                                     self.n_bins,
                                     self.interleave_depth)

    def predict(self, X, majority_vote=False, n_threads=n_threads):

        # print("LET US BEGIN THE GAME")
        # print("No of threads are", n_threads)
        # for i in range(0, len(self.tree_list)):
        #     print("Prediction for tree", i, "original was")
        #     prediction = self.tree_list[i].tree_.predict(np.asarray(X, dtype=np.float32))
        #     # for j in range(0, prediction.shape[0]):
        #     #     print("the orig prediction", j, prediction[j])

        # print("LET US END THE GAME")
        # print("Shape is", X.shape)
        outputs = self._pkd_forest.predict(X, majority_vote, n_threads)
        if majority_vote:
            for i in range(0, outputs.shape[0]):
                print("OUTPUT IS", outputs[i])
            # return self.forest_classifier.estimators_[0].classes_.take(np.max(outputs, axis=1), axis=0)
            # return np.argmax(np.bincount(outputs, axis=1), axis=1)
            # TODO: Apply classes
            a = np.apply_along_axis(np.bincount, axis=1, arr=outputs, minlength = np.max(outputs) +1)
            return np.argmax(a, axis=1)
        else:
            return self.classes_.take(np.argmax(outputs, axis=1), axis=0)
            # return np.argmax(outputs, axis=1)
