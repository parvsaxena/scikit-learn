from libc.stdlib cimport free, malloc

import numpy as np
cimport numpy as np
from ..tree._tree cimport Tree
from ..tree._tree cimport Node

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# Class to represent each bin
cdef class TreeBin:
    cdef public SIZE_t n_trees
    cdef public SIZE_t depth
    cdef public SIZE_t node_count
    cdef Node* nodes         # Only want pkdForest to be able to access it

# TODO: Change the name
cdef class PkdForest:
    # cdef TreeBin* forestRoots
    cdef public SIZE_t n_bins
    cdef public SIZE_t n_trees
    cdef public SIZE_t depth_interleaving