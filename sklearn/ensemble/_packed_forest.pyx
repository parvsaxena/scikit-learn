# distutils: language = c++
from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX
from libcpp.vector cimport vector

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

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t IS_ROOT = 0
cdef SIZE_t IS_LEFT = 1
cdef SIZE_t IS_RIGHT = 2

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
        safe_realloc(&self.working_index, self.n_bins)

        self.node = <PkdNode**> malloc(self.n_bins * sizeof(PkdNode*))


        # Loop to cacl all bins
        for i in range(0, self.n_bins):
            self._create_bin(tree, i)

    # Set bin sizes, and bin offset arrays
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
    # set nodes_per_bin and
    cdef _calc_bin_nodes(self, list trees, SIZE_t bin_no):
        self.n_nodes_per_bin[bin_no] = 0
        for j in range(self.bin_offsets[bin_no], self.bin_offsets[bin_no] + self.bin_sizes[bin_no]):
            self.n_nodes_per_bin[bin_no] += trees[j].node_count

        print("Without classes", self.n_nodes_per_bin[bin_no])
        self.n_nodes_per_bin[bin_no] += trees[self.bin_offsets[bin_no]].max_n_classes
        print("After adding classes", self.n_nodes_per_bin[bin_no])

    cdef _create_bin(self, list trees, SIZE_t bin_no) except +:
        self._calc_bin_nodes(trees, bin_no)
        self.node[bin_no] = <PkdNode*>malloc(self.n_nodes_per_bin[bin_no] * sizeof(PkdNode))
        self.node[bin_no][0].n_node_samples = 1
        print(self.node[bin_no][0].n_node_samples)

        # TODO: Add classes at the end
        # start from size of tree and add classes in bin array
        self._set_classes(trees, bin_no)

        self.working_index[bin_no] = 0
        # set Roots in bin array
        # TODO: Add interleaving support
        print("Copy function comparison")
        for j in range(self.bin_offsets[bin_no], self.bin_offsets[bin_no] + self.bin_sizes[bin_no]):
            #print(trees[j].node_array)
            self._copy_node(&self.node[bin_no][self.working_index[bin_no]], trees[j], 0)
            print(self.node[bin_no][self.working_index[bin_no]].left_child, trees[j].children_left[0])
            self.working_index[bin_no] += 1

        # Create a stack
        cdef vector[NodeRecord] stk

        for j in range(self.bin_offsets[bin_no], self.bin_offsets[bin_no] + self.bin_sizes[bin_no]):
            # TODO: Put assertion back after completing while loop below
            # assert stk.empty()
            # TODO: Assuming root can't be leaf node, crosscheck the assumption
            # Push left or right onto stack
            if self._is_left_child_larger(trees[j], 0):
                # push right, then left
                stk.push_back(NodeRecord(j, trees[j].children_right[0], IS_RIGHT, j - self.bin_offsets[bin_no], 1))
                stk.push_back(NodeRecord(j, trees[j].children_left[0], IS_LEFT, j - self.bin_offsets[bin_no], 1))
            else:
                NodeRecord(j, trees[j].children_right[0], IS_RIGHT, j - self.bin_offsets[bin_no], 1)
                stk.push_back(NodeRecord(j, trees[j].children_left[0], IS_LEFT, j - self.bin_offsets[bin_no], 1))
                stk.push_back(NodeRecord(j, trees[j].children_right[0], IS_RIGHT, j - self.bin_offsets[bin_no], 1))

            # while stk not empty, keep going
            while not stk.empty():
                self._process_node(stk.back(), stk, trees, bin_no)
                # process it

    # Copy node from tree
    cdef _copy_node(self, PkdNode* pkdNode, object tree, SIZE_t node_id):
        # print(tree.children_left[node_id])
        pkdNode.left_child = tree.children_left[node_id]
        pkdNode.right_child = tree.children_right[node_id]
        pkdNode.feature = tree.feature[node_id]
        pkdNode.threshold = tree.threshold[node_id]
        pkdNode.n_node_samples = tree.n_node_samples[node_id]

    # Copy node from NodeRecord
    cdef _copy_processed_node(self, PkdNode *pkdNode, NodeRecord &node, SIZE_t working_index, list trees):
        pkdNode.left_child = trees[node.tree_id].children_left[node.node_id]
        pkdNode.right_child = trees[node.tree_id].children_right[node.node_id]
        pkdNode.feature = trees[node.tree_id].feature[node.node_id]
        pkdNode.threshold = trees[node.tree_id].threshold[node.node_id]
        pkdNode.n_node_samples = trees[node.tree_id].n_node_samples[node.node_id]

    cdef _link_parent_to_node(self, PkdNode *pkdNode_p, SIZE_t working_index, NodeRecord &node):
        # link the bin position of node to parent's left or right
        if node.node_type == IS_LEFT:
            pkdNode_p.left_child = working_index
        elif node.node_type == IS_RIGHT:
            pkdNode_p.right_child = working_index

    cdef bint _is_leaf(self, NodeRecord &node, object tree):
        return tree.children_left[node.node_id] == _TREE_LEAF

    cdef bint _is_internal_node(self, NodeRecord &node, object tree):
        return tree.children_left[node.node_id] != _TREE_LEAF

    cdef bint _is_left_child_larger(self, object tree, SIZE_t node_id):
        return tree.n_node_samples[tree.children_left[node_id]] > tree.n_node_samples[tree.children_right[node_id]]

    cdef _process_node(self, NodeRecord &node, vector[NodeRecord] &stk, list trees, SIZE_t bin_no):
        print("Node ID is", node.node_id)
        stk.pop_back()
        if self._is_leaf(node, trees[node.tree_id]):
            # dummy stmt
            abc = 1
            # Copy processed node to bin
            # Link Parent to child
            # Increment working_index
        else:
            # dummy stmt
            abc = 2
            # Copy processed node to bin
            self._copy_processed_node(&self.node[bin_no][self.working_index[bin_no]], node, self.working_index[bin_no], trees)
            # Link Parent to child
            self._link_parent_to_node(&self.node[bin_no][self.working_index[bin_no]], self.working_index[bin_no], node)
            # Increment working index
            self.working_index[bin_no] += 1
            # create and push child nodes

    cdef _set_classes(self, list trees, SIZE_t bin_no):
        print("Total nodes are ", self.n_nodes_per_bin[bin_no])
        print("No nodes w/o classes are ", self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes)
        print("Total classes are", trees[self.bin_offsets[bin_no]].max_n_classes)
        for i in range(0, trees[self.bin_offsets[bin_no]].max_n_classes):
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].left_child = _TREE_LEAF
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].right_child = i
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].feature = _TREE_UNDEFINED
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].threshold = _TREE_UNDEFINED
            print("node no ", self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i)
            print("was assigned ", i)