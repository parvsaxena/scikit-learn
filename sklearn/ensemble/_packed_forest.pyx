# cython: profile=True
# distutils: language = c++
from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX
from libcpp.vector cimport vector
from libcpp.deque cimport deque
from cython.parallel import prange
IF SKLEARN_OPENMP_PARALLELISM_ENABLED:
    cimport openmp

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
cdef SIZE_t IS_ROOT = 2
cdef SIZE_t IS_LEFT = 0
cdef SIZE_t IS_RIGHT = 1

cdef class PkdForest:
    def __cinit__(self, list tree, SIZE_t n_bins, SIZE_t depth_interleaving):

        # TODO: This is unoptimized
        # as not knowing type beforehand hampers while passing list
        # cython performance, find workaround, if possible??

        self.n_trees = len(tree)
        self.n_bins = n_bins
        self.depth_interleaving = depth_interleaving
        self.max_n_classes = tree[0].max_n_classes

        # print("No of trees, first tree size, interleave_depth and bins are")
        # print( self.n_trees,
        #       tree[0].node_count,
        #       self.depth_interleaving,
        #       self.n_bins )

        self._calc_bin_sizes()

        safe_realloc(&self.n_nodes_per_bin, self.n_bins)
        safe_realloc(&self.working_index, self.n_bins)

        self.node = <PkdNode**> malloc(self.n_bins * sizeof(PkdNode*))

        # Loop to calc nodes in bins
        for i in range(0, self.n_bins):
            self._calc_bin_nodes(tree, i)

        max_nodes_across_bin = self._max_nodes_across_bin()
        self.value = np.zeros(shape=(self.n_bins, max_nodes_across_bin, 2, self.max_n_classes), dtype=np.float64)

        # Loop to calc all bins
        for i in range(0, self.n_bins):
            self._create_bin(tree, i)


    cdef SIZE_t _max_nodes_across_bin(self):
        max_nodes = 0
        for i in range(0, self.n_bins):
            if self.n_nodes_per_bin[i] > max_nodes:
                max_nodes = self.n_nodes_per_bin[i]
        # print("Max nodes across bins ", max_nodes)
        return max_nodes

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

        # for i in range(0, self.n_bins):
        #     print("BIN SIZES, BIN_OFFSETS")
        #     print(self.bin_sizes[i], self.bin_offsets[i])

    # TODO: Add check to ensure no_of_unique_classes is same for all forest
    # set nodes_per_bin and
    cdef _calc_bin_nodes(self, list trees, SIZE_t bin_no):
        self.n_nodes_per_bin[bin_no] = 0
        for j in range(self.bin_offsets[bin_no], self.bin_offsets[bin_no] + self.bin_sizes[bin_no]):
            self.n_nodes_per_bin[bin_no] += trees[j].node_count

        # print("Without classes", self.n_nodes_per_bin[bin_no])
        self.n_nodes_per_bin[bin_no] += trees[self.bin_offsets[bin_no]].max_n_classes
        # print("After adding classes", self.n_nodes_per_bin[bin_no])

    cdef _create_bin(self, list trees, SIZE_t bin_no) except +:
        self.node[bin_no] = <PkdNode*>malloc(self.n_nodes_per_bin[bin_no] * sizeof(PkdNode))
        #self.node[bin_no][0].n_node_samples = 1
        # print(self.node[bin_no][0].n_node_samples)

        # TODO: Add classes at the end
        # start from size of tree and add classes in bin array
        self._set_classes(trees, bin_no)

        # Create a deque
        cdef deque[NodeRecord] deq

        self.working_index[bin_no] = 0
        # set Roots in bin array
        # TODO: Add interleaving support
        # print("Copy function comparison")
        for j in range(self.bin_offsets[bin_no], self.bin_offsets[bin_no] + self.bin_sizes[bin_no]):
            # print(trees[j].node_array)
            # print("Trees")
            # print(trees[j].children_right)
            # print(trees[j].children_left)
            # print(trees[j].feature)
            # print(trees[j].threshold)
            # print(trees[j].n_node_samples)
            # TODO 1: Change to push_back
            deq.push_back(NodeRecord(j, 0, IS_ROOT, -1, 0))
            # print(trees[deq.back().tree_id].children_left[deq.back().node_id], trees[j].children_left[0])
            #self._copy_node(&self.node[bin_no][self.working_index[bin_no]], trees[j], 0)
            # print(self.node[bin_no][self.working_index[bin_no]].left_child, trees[j].children_left[0])
            #self.working_index[bin_no] += 1
        '''
        for j in range(self.bin_offsets[bin_no], self.bin_offsets[bin_no] + self.bin_sizes[bin_no]):
            # TODO: Put assertion back after completing while loop below
            assert deq.empty()
            # TODO: Assuming root can't be leaf node, crosscheck the assumption
            # Push left or right onto stack
            if self._is_left_child_larger(trees[j], 0):
                # push right, then left
                deq.push_back(NodeRecord(j, trees[j].children_right[0], IS_RIGHT, j - self.bin_offsets[bin_no], 1))
                deq.push_back(NodeRecord(j, trees[j].children_left[0], IS_LEFT, j - self.bin_offsets[bin_no], 1))
            else:
                NodeRecord(j, trees[j].children_right[0], IS_RIGHT, j - self.bin_offsets[bin_no], 1)
                deq.push_back(NodeRecord(j, trees[j].children_left[0], IS_LEFT, j - self.bin_offsets[bin_no], 1))
                deq.push_back(NodeRecord(j, trees[j].children_right[0], IS_RIGHT, j - self.bin_offsets[bin_no], 1))

            # while deq not empty, keep going
            while not deq.empty():
                self._process_node(deq.back(), deq, trees, bin_no)
                # process it
        '''

        while not deq.empty():
            if deq.front().depth > self.depth_interleaving:
                # print("Calling process node with False")
                self._process_node(deq.front(), deq, trees, bin_no, False)
            else:
                # print("Calling process node with True")
                self._process_node(deq.front(), deq, trees, bin_no, True)

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
        else:
            print("Case not handled yet")

    cdef bint _is_leaf(self, NodeRecord &node, object tree):
        return tree.children_left[node.node_id] == _TREE_LEAF

    cdef bint _is_internal_node(self, NodeRecord &node, object tree):
        return tree.children_left[node.node_id] != _TREE_LEAF

    cdef bint _is_left_child_larger(self, object tree, SIZE_t node_id):
        return tree.n_node_samples[tree.children_left[node_id]] > tree.n_node_samples[tree.children_right[node_id]]

    cdef bint _is_root_node(self, NodeRecord &node):
        return node.node_type == IS_ROOT

    cdef _process_leaf_node(self, list trees, NodeRecord &node, SIZE_t bin_no):
        # print("Going into leaf")

        # Find max class in value
        # print("Shape of value array is ", trees[node.tree_id].value.shape)
        node_class_label = np.argmax(trees[node.tree_id].value[node.node_id][0])

        # TODO: Add a check in python function to make sure n_outputs = 1
        # Second value in this is 0 for that reason
        # print("Array is ", trees[node.tree_id].value[node.node_id][0])
        # print("node class ", node_class_label)

        # Link Parent to leaf class
        # TODO 2: Only link if not root
        if not self._is_root_node(node):
            self._link_parent_to_node(&self.node[bin_no][node.parent_id], self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + node_class_label, node)

        cdef DOUBLE_t[:] value_array = trees[node.tree_id].value[node.node_id][0]/np.sum(trees[node.tree_id].value[node.node_id][0])

        if node.node_type == IS_LEFT:
            self.value[bin_no][node.parent_id][IS_LEFT] = value_array
            # print("Array after processing ", np.asarray(self.value[bin_no][node.parent_id][IS_LEFT]))
        else:
            self.value[bin_no][node.parent_id][IS_RIGHT] = value_array
            # print("Array after processing ", np.asarray(self.value[bin_no][node.parent_id][IS_RIGHT]))
        # Increment working_index - NO NEED???

    cdef _process_internal_node(self, list trees, NodeRecord &node, SIZE_t bin_no, deque[NodeRecord] &deq, bint interleave):
        # print("Going into non-leaf")

        # Copy processed node to bin
        self._copy_processed_node(&self.node[bin_no][self.working_index[bin_no]], node, self.working_index[bin_no], trees)

        # Link Parent to child
        # TODO 2: Only link if not root
        if not self._is_root_node(node):
            self._link_parent_to_node(&self.node[bin_no][node.parent_id], self.working_index[bin_no], node)

        # BEHAVIOR FOR interleave = False
        # create and push child nodes
        if interleave == False:
            if self._is_left_child_larger(trees[node.tree_id], node.node_id):
                # Push right child, then left
                # print("Pushing left greater with interleave False")
                deq.push_front(NodeRecord(node.tree_id, trees[node.tree_id].children_right[node.node_id], IS_RIGHT, self.working_index[bin_no], node.depth + 1))
                deq.push_front(NodeRecord(node.tree_id, trees[node.tree_id].children_left[node.node_id], IS_LEFT, self.working_index[bin_no], node.depth + 1))
            else:
                # Push left child, then right
                # print("Pushing right greater with interleave False")
                deq.push_front(NodeRecord(node.tree_id, trees[node.tree_id].children_left[node.node_id], IS_LEFT, self.working_index[bin_no], node.depth + 1))
                deq.push_front(NodeRecord(node.tree_id, trees[node.tree_id].children_right[node.node_id], IS_RIGHT, self.working_index[bin_no], node.depth + 1))

        else:
            if node.depth == self.depth_interleaving:
                # do stat, but still push back
                if self._is_left_child_larger(trees[node.tree_id], node.node_id):
                    # Push left child, then right
                    # print("Pushing left greater with node.depth = interleave depth")
                    deq.push_back(NodeRecord(node.tree_id, trees[node.tree_id].children_left[node.node_id], IS_LEFT, self.working_index[bin_no], node.depth + 1))
                    deq.push_back(NodeRecord(node.tree_id, trees[node.tree_id].children_right[node.node_id], IS_RIGHT, self.working_index[bin_no], node.depth + 1))
                else:
                    # Push right child, then left
                    # print("Pushing right greater with node.depth = interleave depth")
                    deq.push_back(NodeRecord(node.tree_id, trees[node.tree_id].children_right[node.node_id], IS_RIGHT, self.working_index[bin_no], node.depth + 1))
                    deq.push_back(NodeRecord(node.tree_id, trees[node.tree_id].children_left[node.node_id], IS_LEFT, self.working_index[bin_no], node.depth + 1))
            else:
                # push L, push R, at back
                # print("LR pushing wtih interleave True")
                deq.push_back(NodeRecord(node.tree_id, trees[node.tree_id].children_left[node.node_id], IS_LEFT, self.working_index[bin_no], node.depth + 1))
                deq.push_back(NodeRecord(node.tree_id, trees[node.tree_id].children_right[node.node_id], IS_RIGHT, self.working_index[bin_no], node.depth + 1))

        # print(node.node_id, self.working_index[bin_no])
        # Increment working index
        self.working_index[bin_no] += 1

    cdef _process_node(self, NodeRecord node, deque[NodeRecord] &deq, list trees, SIZE_t bin_no, bint interleave):
        # print("Node ID is", node.node_id)
        deq.pop_front()

        if self._is_leaf(node, trees[node.tree_id]):
            self._process_leaf_node(trees, node, bin_no)

        else:
            self._process_internal_node(trees, node, bin_no, deq, interleave)

    cdef _set_classes(self, list trees, SIZE_t bin_no):
        # print("Total nodes are ", self.n_nodes_per_bin[bin_no])
        # print("No nodes w/o classes are ", self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes)
        # print("Total classes are", trees[self.bin_offsets[bin_no]].max_n_classes)
        for i in range(0, trees[self.bin_offsets[bin_no]].max_n_classes):
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].left_child = _TREE_LEAF
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].right_child = i
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].feature = _TREE_UNDEFINED
            self.node[bin_no][self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i].threshold = _TREE_UNDEFINED
            # print("node no ", self.n_nodes_per_bin[bin_no] - trees[self.bin_offsets[bin_no]].max_n_classes + i)
            # print("was assigned ", i)

    cpdef np.ndarray predict(self, object X_ndarray, bint majority_vote, SIZE_t n_threads):
        cdef const DOUBLE_t[:, :] X = X_ndarray
        # IF SKLEARN_OPENMP_PARALLELISM_ENABLED:
        #     print(openmp.omp_get_max_threads())

        # Create loop for all observations
        # TODO: NOTE, max_n_classes no assigned any value in code
        cdef SIZE_t[:,:] predict_array = np.zeros(shape=(X.shape[0], self.n_trees), dtype=np.intp)
        cdef DOUBLE_t[:,:,:] predict_matrix = np.zeros(shape=(X.shape[0], self.n_trees, self.max_n_classes), dtype=np.float64)
        # see bin size of 1st bin as it will be maximum, and differ by 1 tree from others(at max)
        cdef SIZE_t[:,:] curr_node = np.zeros(shape=(self.n_bins, self.bin_sizes[0]), dtype=np.intp)
        # print("Curr_node shape is", curr_node.shape, curr_node.ndim)
        # print("Predict shape is", predict_array.shape, predict_array.ndim)
        # TODO: Add parallel support here

        cdef:
            SIZE_t bin_no, obs_no, tree_no
            SIZE_t internal_nodes_reached
            SIZE_t next_node, child

        for bin_no in prange(0, self.n_bins, nogil=True, schedule='static', num_threads = n_threads):
            # print("STarting code for bin no", bin_no)

            for obs_no in range(0, X.shape[0]):
                # print("observation no ", obs_no)
                # print("observation is ", X[obs_no])

                # Initialize to tree roots in the bins
                for tree_no in range(0, self.bin_sizes[bin_no]):
                    curr_node[bin_no][tree_no] = tree_no

                # print("Curr node after initialization is", np.asarray(curr_node[bin_no,:]))

                internal_nodes_reached = self.bin_sizes[bin_no]
                while internal_nodes_reached > 0:
                    internal_nodes_reached = 0
                    # print("Curr node after initialization is ", np.asarray(curr_node[bin_no,:]))
                    for tree_no in range(0, self.bin_sizes[bin_no]):
                        # print("Tree no", tree_no)
                        # print("Printing node", self.node[bin_no][curr_node[bin_no][tree_no]])
                        if not self._is_class_node(&self.node[bin_no][curr_node[bin_no][tree_no]]):
                            next_node, child = self._find_next_node(&self.node[bin_no][curr_node[bin_no,tree_no]], obs_no, X)
                            if self._is_class_node(&self.node[bin_no][next_node]):
                                # print("Class node found!!")
                                predict_matrix[obs_no, tree_no + self.bin_offsets[bin_no]]= self.value[bin_no][curr_node[bin_no][tree_no]][child]
                                # print("Writing for original tree no ", tree_no + self.bin_offsets[bin_no])
                            curr_node[bin_no,tree_no] = next_node
                            # print("Next node and child are", next_node, child)
                            # print("Predict matrix here is ", np.asarray(predict_matrix[obs_no,tree_no + self.bin_offsets[bin_no]]))
                            internal_nodes_reached = internal_nodes_reached + 1
                            # print("current node now is", curr_node[bin_no,tree_no])

                # time to predict classes
                if majority_vote == True:
                    for tree_no in range(0, self.bin_sizes[bin_no]):
                        predict_array[obs_no, self.bin_offsets[bin_no] + tree_no] = self.node[bin_no][curr_node[bin_no,tree_no]].right_child


            # print("Prediction internally is", np.asarray(predict_array[obs_no,:]))

        if majority_vote == False:
            # prediction by average
            # print("Avg probabilities are")
            array = np.mean(predict_matrix, axis=1)
            # for i in range(0, array.shape[0]):
            #     print("Before avg is", i, np.asarray(predict_matrix[i]))
            #     print("Average prediction", i, array[i])

            return array
        else:
            return np.asarray(predict_array, dtype = np.intp)

    cdef bint _is_class_node(self, PkdNode* pkdNode) nogil:
        return pkdNode.left_child == _TREE_LEAF

    # TODO: Avoid passing object X, pass reference
    cdef (SIZE_t, SIZE_t) _find_next_node(self, PkdNode* pkdNode, SIZE_t obs_no, const DOUBLE_t[:,:] X) nogil:
        # TODO: Make sure this pkdNode is not class node
        if(X[obs_no][pkdNode.feature] <= pkdNode.threshold):
            return pkdNode.left_child, IS_LEFT
        else:
            return pkdNode.right_child, IS_RIGHT
