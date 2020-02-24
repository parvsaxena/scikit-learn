from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
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
    def __cinit__(self, list tree, SIZE_t tree_size):
        cdef abc = 1
        # print tree[0]
        print(abc)
        #print(tree[0].n_features)

        # TODO: This is unoptimized
        # as not knowing type beforehand hampers
        # cython performance, find workaround, if possible??

        print(tree[0].node_count)
