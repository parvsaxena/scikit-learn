from libc.stdlib cimport free, malloc

import numpy as np
cimport numpy as np
from ..tree._tree cimport Tree
from ..tree._tree cimport Node

ctypedef np.npy_float32 DTYPE_t             # Type of X
ctypedef np.npy_float64 DOUBLE_t            # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t                 # Type for indices and counters
ctypedef np.npy_int32 INT32_t               # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t             # Unsigned 32 bit integer

cdef struct PkdNode:
    # Base storage structure for the nodes in a Packed Forest
    SIZE_t left_child                       # id of the left child of the node
    SIZE_t right_child                      # id of the right child of the node
    SIZE_t feature                          # Feature used for splitting the node
    DOUBLE_t threshold                      # Threshold value at the node
    SIZE_t n_node_samples                   # Number of samples at the node
    SIZE_t depth                            # Depth of the node in Tree

# TODO: Change the name
cdef class PkdForest:
    cdef public SIZE_t n_bins               # No of bins
    cdef public SIZE_t n_trees              # Total no of trees in forest
    cdef public SIZE_t depth_interleaving   # Tree depth to interleave in bin
    cdef PkdNode** node                     # 2-D array of nodes, where first one is leaf
    cdef SIZE_t* n_nodes_per_bin            # No of nodes in bin []
    cdef SIZE_t* bin_sizes                  # No of trees in bin []
    cdef SIZE_t* bin_offsets                # Tree offsets for bins
    # Methods
    cdef _calc_bin_sizes(self)
    cdef _calc_bin_nodes(self, list tree, SIZE_t bin_no)