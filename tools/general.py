import numpy as np


def remove_duplicate_nodes(nodes, elems, stable_order=True, return_index=False, return_inverse=False):
    _, ind, ind_inv = np.unique(nodes, axis=0, return_index=True, return_inverse=True)

    # maintain old order of nodes
    if stable_order:
        id = np.argsort(ind)
        ind = ind[id]
        id0 = np.empty((id.size,),dtype=int)
        id0[id] = np.arange(id.size)
        ind_inv = id0[ind_inv]

    nodes = nodes[ind,:]
    elems = ind_inv[elems]

    return_tup = (nodes, elems)
    if return_index:
        return_tup += (ind,)
    if return_inverse:
        return_tup += (ind_inv,)

    return return_tup
