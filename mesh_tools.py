import re, csv
import numpy as np


def read_inp(file):
    with open(file,'r') as f:
        s = f.read()
    match = re.search(r'\*.*NODE[\S ]*\s+(.*)\*.*END NODE.*\*.*ELEMENT[\S ]*\s+(.*)\*.*END ELEMENT', s, re.MULTILINE | re.DOTALL)
    nodes, elems = match.group(1), match.group(2)

    # print(nodes,elems)

    nodes = np.array(list(csv.reader(nodes.strip().split('\n'))), dtype=float)[:,1:]
    elems = np.array(list(csv.reader(elems.strip().split('\n'))), dtype=int)[:,1:] - 1
    return nodes, elems


def seed():
    return np.array([
        [+1,+1,+1],
        [+1,-1,+1],
        [-1,-1,+1],
        [-1,+1,+1],
        [+1,+1,-1],
        [+1,-1,-1],
        [-1,-1,-1],
        [-1,+1,-1],
    ]) * -1


def calculate_grid(nodes, elems):
    grid_config = np.array([
        [1,1,0,0,1,1,0,0],
        [1,0,0,1,1,0,0,1],
        [1,1,1,1,0,0,0,0]]).T
    cnfg = grid_config - grid_config.min(axis=0)
    node_grid = np.zeros(nodes.shape)
    node_grid[:] = np.nan
    node_grid[elems[0,0],:] = 0
    ind_add = elems[0,0]
    ele = np.unique(elems)
    ind_unset = np.arange(nodes.shape[0])
    while np.intersect1d(ind_unset,ele).any():
        ind_unset = np.setdiff1d(ind_unset, ind_add)
        r,c = np.nonzero(np.isin(elems, ind_add))
        sub_ele = elems[r,:]
        ind_add, id, *_ = np.intersect1d(sub_ele.flatten(), ind_unset, return_indices=True)
        sub_r, sub_c = np.unravel_index(id, sub_ele.shape)
        node_grid[sub_ele[(sub_r, sub_c)],:] = node_grid[sub_ele[(sub_r, c[sub_r])],:] + cnfg[sub_c,:] - cnfg[c[sub_r],:]
    
    node_grid = node_grid - node_grid.min(axis=0)
    # elem_grid = node_grid[elems.T,:].mean(axis=0)
    return node_grid.astype(int)



def grid_3d(node_grid):
    g3d = -np.ones(node_grid.max(axis=0)+1, dtype=int)
    g3d[(*node_grid.T,)] = np.arange(node_grid.shape[0])
    # for i,ng in enumerate(node_grid):
    #     print(ng)
    #     g3d[*ng] = i
    return g3d


def seed_grid(nodes, elems):
    N, E = nodes, elems
    ng = np.zeros_like(N)
    # assumes u,v,w correspond to x,y,z, in both dimension and direction
    # first, find 8 nodes with single occurence
    occr = np.bincount(E.flat)
    n8 = np.where(occr==1)[0]
    cols = []
    for n in n8:
        cols.append(np.mgrid[range(E.shape[0]), range(E.shape[1])][1][np.isin(E,n)][0])
    n8[cols] = n8

    assert n8.size==8, 'check mesh'
    # then, set the grid position of these eight nodes
    # set left and right (-x -> -INF, +x -> +INF)
    ng[n8,0] = np.where(N[n8,0]<np.median(N[n8,0]), np.NINF, np.PINF)
    # set up and down (-z -> -INF, +z -> +INF)
    ng[n8,2] = np.where(N[n8,2]<np.median(N[n8,2]), np.NINF, np.PINF)
    # set front and back
    n4 = n8[N[n8,2]<np.median(N[n8,2])] # top 4
    c = N[n4].mean(axis=0)
    n2 = n4[N[n4,0]<np.median(N[n4,0])] # top left
    d = np.sum((N[n2] - c)**2, axis=1)**.5
    ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
    n2 = n4[N[n4,0]>np.median(N[n4,0])] # top right
    d = np.sum((N[n2] - c)**2, axis=1)**.5
    ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

    n4 = n8[N[n8,2]>np.median(N[n8,2])] # bottom 4
    c = N[n4].mean(axis=0)
    n2 = n4[N[n4,0]<np.median(N[n4,0])] # bottom left
    d = np.sum((N[n2] - c)**2, axis=1)**.5
    ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
    n2 = n4[N[n4,0]>np.median(N[n4,0])] # bottom right
    d = np.sum((N[n2] - c)**2, axis=1)**.5
    ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

    seed = np.ones((8,3))*.5
    ind_preset = np.all(np.isinf(ng), axis=1).nonzero()[0]
    for row,col in zip(*np.where(np.isin(N, ind_preset))):
        seed[col] *= np.sign(ng[N[row,col]])
    return seed


def lip_node_index(node_grid):
    _, ind = np.unique(node_grid, axis=0, return_index=True)
    ind_1f = np.setdiff1d(np.arange(node_grid.shape[0]), ind)
    z_ind = np.unique(node_grid[ind_1f,2])
    assert len(z_ind)==1, 'check mesh'
    z_ind = z_ind[0]
    ind = np.isin(node_grid[:,0], node_grid[ind_1f,0]) & (node_grid[:,2] == z_ind)
    ind = np.unique(ind.nonzero()[0])
    assert len(ind)%6 == 0, 'check_mesh'
    return ind


if __name__=='__main__':
    
    nodes, elems = read_inp(r'C:\Users\tians\Downloads\n0034\hexmesh.inp')
    node_grid = calculate_grid(nodes, elems)
    lip_index = lip_node_index(node_grid)
    import scipy
    scipy.io.loadmat(r'C:\Users\tians\Downloads\n0034\lip_node_index.mat')
    from sklearn.neighbors import NearestNeighbors
    # nbrs = NearestNeighbors(n_neighbors=1).fit(node_grid)
    # _, index = nbrs.kneighbors(node_grid[lip_index,:])
    ind = np.argsort(node_grid[lip_index,2])
    lip_index = lip_index[ind]
    
    ind = np.argsort(node_grid[lip_index,1])
    lip_index = lip_index[ind]
    
    print(lip_index.reshape(6,-1))
