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

__all__ = ["PackedForest"]


class PackedForest:
    def __init__(self,
                 interleave_depth=1,
                 n_bins=128,
                 forest_classifier=None):

        self.interleave_depth = interleave_depth
        self.n_bins = n_bins
        self.forest_classifier = forest_classifier
        print(forest_classifier.estimators_)
        self.n_trees = forest_classifier.n_estimators
        self.tree_list = forest_classifier.estimators_
        for tree in self.tree_list:
            print(tree.tree_.node_count, tree.tree_.capacity)
        print(self.tree_list)

        PkdForest([tree.tree_ for tree in self.tree_list], len(self.tree_list))

    def predict(self,
                x):
        pass
