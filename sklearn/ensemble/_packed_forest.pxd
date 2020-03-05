from libc.stdlib cimport free, malloc
from libcpp.vector cimport vector

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
    SIZE_t left_child                       # Id of the left child of the node
    SIZE_t right_child                      # Id of the right child of the node
    SIZE_t feature                          # Feature used for splitting the node
    DOUBLE_t threshold                      # Threshold value at the node
    SIZE_t n_node_samples                   # Number of samples at the node

cdef struct NodeRecord:
    # Aux data structure used for pushing into stack and queue
    SIZE_t tree_id                          # tree_id in forest
    SIZE_t node_id                          # node_id in tree original Tree
    SIZE_t node_type                        # 0 = root, 1 = left, 2 = right
    SIZE_t parent_id                        # position of parent in bin array
    SIZE_t depth                            # depth of node in original Tree

# TODO: Change the name
cdef class PkdForest:
    cdef public SIZE_t n_bins               # No of bins
    cdef public SIZE_t n_trees              # Total no of trees in forest
    cdef public SIZE_t depth_interleaving   # Tree depth to interleave in bin
    cdef PkdNode** node                     # 2-D array of nodes, where first one is leaf
    cdef SIZE_t* n_nodes_per_bin            # No of nodes in bin []
    cdef SIZE_t* bin_sizes                  # No of trees in bin []
    cdef SIZE_t* bin_offsets                # Tree offsets for bins
    cdef SIZE_t* working_index              # working index for node in bin []
    # TODO: Assign the variable and replace
    #cdef SIZE_t  max_n_classes              # Maximum no of classes in forest

    # Methods
    cdef _calc_bin_sizes(self)
    cdef _calc_bin_nodes(self, list trees, SIZE_t bin_no)
    cdef _create_bin(self, list trees, SIZE_t bin_no) except +
    cdef _copy_node(self, PkdNode* pkdNode, object node, SIZE_t node_id)
    cdef bint _is_leaf(self, NodeRecord &node, object tree)
    cdef bint _is_class_node(self, PkdNode* pkdNode)
    cdef bint _is_internal_node(self, NodeRecord &node, object tree)
    cdef _process_node(self, NodeRecord node, vector[NodeRecord] &stk, list trees, SIZE_t bin_no)
    cdef bint _is_left_child_larger(self, object tree, SIZE_t node_id)
    cdef _set_classes(self, list trees, SIZE_t bin_no)
    cdef _copy_processed_node(self, PkdNode *pkdNode, NodeRecord &node, SIZE_t working_index, list trees)
    cdef _link_parent_to_node(self, PkdNode *pkdNode_p, SIZE_t working_index, NodeRecord &node)
    cpdef np.ndarray predict(self, object X)
    cdef SIZE_t _find_next_node(self, PkdNode* pkdNode, SIZE_t obs_no, object X)