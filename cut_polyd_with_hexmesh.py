import os
from scipy.interpolate import *
from general_tools import *
from mesh_tools import *
from polydata_tools import *
from rendering_tools import *
from image_tools import *




if __name__ == '__main__':
    os.chdir(r'C:\data\20230501\n0034')
    nodes, elems = read_inp(r'C:\data\20230501\n0034\hexmesh.inp')
    node_grid, ind_upper, ind_lwoer = calculate_grid(nodes, elems, calculate_lip_index=True)

    nodes, elems = read_inp(r'hexmesh_open.inp')
    nodes = np.ascontiguousarray(nodes)
    elems = np.ascontiguousarray(elems)
    N = nodes.copy()
    X,Y,Z = N[:,0], N[:,1], N[:,2]
    
    node_grid, ind_upper, ind_lower = calculate_grid(nodes, elems, calculate_lip_index=True)


    # regular grid by mean lip
    nodes_rg = nodes
    lip_mean = nodes_rg[ind_upper,:]/2 + nodes_rg[ind_lower,:]/2
    nodes_rg[ind_upper,:] = lip_mean
    nodes_rg[ind_lower,:] = lip_mean
    _, ind, ind_inv = np.unique(nodes_rg, axis=0, return_index=True, return_inverse=True)
    # id = np.argsort(ind)
    # ind = ind[id]
    # id0 = np.empty((id.size,),dtype=int)
    # id0[id] = np.arange(id.size)
    # ind_inv = id0[ind_inv]
    elems_rg = ind_inv[elems]
    nodes_rg = nodes_rg[ind,:]
    node_grid_rg = node_grid[ind,:]
    X,Y,Z = nodes_rg[:,0], nodes_rg[:,1], nodes_rg[:,2]
    g3d = grid_3d_from_flat(node_grid_rg)
    node_grid_rg_recover = grid_flat_from_3d(g3d)
    print(np.all(node_grid_rg_recover == node_grid_rg))
    elems_recover = elements_from_grid_3d(g3d, seed=None)
    print(np.all(elems_recover[np.argsort(elems_recover[:,0])] == elems_rg[np.argsort(elems_rg[:,0])]))

    # extrapolate to expand mesh

    G = np.mgrid[
        -2:node_grid_rg[:,0].max()+3,
        -2:node_grid_rg[:,1].max()+3,
        -2:node_grid_rg[:,2].max()+3,
    ]
    g = np.concatenate(G).reshape(3,-1).T.astype(float)

    # g_keep = g3d[:,:3,:]
    # ind_find = g[:,1]<0
    # g[ind_find,:] = RBFInterpolator(node_grid_rg[g_keep.flat,:], nodes_rg[g_keep.flat,:], degree=3)(g[ind_find,:])
    # print('yes')

    g_keep = g3d[:5,:,:]
    ind_find = g[:,0]<0
    g[ind_find,:] = RBFInterpolator(node_grid_rg[g_keep.flat,:], nodes_rg[g_keep.flat,:], degree=3)(g[ind_find,:])

    g_keep = g3d[-5:, :, :]
    ind_find = g[:, 0] > node_grid_rg[:, 0].max()
    g[ind_find, :] = RBFInterpolator(
        node_grid_rg[g_keep.flat, :], nodes_rg[g_keep.flat, :], degree=3)(g[ind_find, :])

    bone_nodes = np.nonzero(node_grid[:,1]==node_grid[:,1].max())
    faces = elems[np.isin(elems, bone_nodes).any(axis=1)]
    faces = np.ascontiguousarray(faces[:,[0,3,7,4]], dtype=np.int64)
