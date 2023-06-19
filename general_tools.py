import numpy as np


def remove_duplicate_nodes(nodes, elems, return_index=False, return_inverse=False):
    nodes, ind = np.unique(nodes, axis=0, return_index=return_index, return_inverse=return_inverse)
    elems = ind[elems]
    return nodes, elems, ind
