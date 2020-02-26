from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from ..tree._utils cimport Stack
from ..tree._utils cimport StackRecord
from ..tree._utils cimport safe_realloc
from ..tree._utils cimport sizet_ptr_to_ndarray
from ..tree._tree cimport Tree

cdef class PkdForest:
    def __cinit__(self, list tree, SIZE_t n_bins, SIZE_t depth_interleaving):

        # TODO: This is unoptimized
        # as not knowing type beforehand hampers while passing list
        # cython performance, find workaround, if possible??

        self.n_trees = len(tree)
        self.n_bins = n_bins
        self.depth_interleaving = depth_interleaving

        print("No of trees, first tree size, interleave_depth and bins are")
        print( self.n_trees,
               tree[0].node_count,
               self.depth_interleaving,
               self.n_bins )

        self._calc_bin_sizes()

        safe_realloc(&self.n_nodes_per_bin, self.n_bins)
        # Loop to cacl all bins
        for i in range(0, self.n_bins):
            self._calc_bin_nodes(tree, i)


    cdef _calc_bin_sizes(self):
        cdef SIZE_t min_bin_size = <SIZE_t>(self.n_trees/self.n_bins)
        safe_realloc(&self.bin_sizes, self.n_bins)
        safe_realloc(&self.bin_offsets, self.n_bins)
        cdef remaining_trees = self.n_trees - min_bin_size*self.n_bins

        for i in range(0, self.n_bins):
            self.bin_offsets[i] = 0
            self.bin_sizes[i] = min_bin_size
            # Distribute remaining trees
            if remaining_trees > 0:
                remaining_trees -= 1
                self.bin_sizes[i] += 1

            # Calculate trees offsets for bin
            if i != 0:
                self.bin_offsets[i] = self.bin_offsets[i-1] + self.bin_sizes[i-1]



        for i in range(0, self.n_bins):
            print(self.bin_sizes[i], self.bin_offsets[i])

    # TODO: Add check to ensure no_of_unique_classes is same for all forest
    cdef _calc_bin_nodes(self, list tree, SIZE_t bin_no):
        self.n_nodes_per_bin[bin_no] = 0
        for j in range(self.bin_offsets[bin_no], self.bin_offsets[bin_no] + self.bin_sizes[bin_no]):
            self.n_nodes_per_bin[bin_no] += tree[j].node_count

        print("Without classes", self.n_nodes_per_bin[bin_no])
        self.n_nodes_per_bin[bin_no] += tree[self.bin_offsets[bin_no]].max_n_classes
        print("After adding classes", self.n_nodes_per_bin[bin_no])
