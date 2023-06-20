import os
from scipy.interpolate import *
from scipy.interpolate import RBFInterpolator
from general_tools import *
from mesh_tools import *
from polydata_tools import *
from rendering_tools import *
from image_tools import *

def bt(a:numpy.ndarray,lo,hi):
    return a>=lo & a<hi

if __name__ == '__main__':
    os.chdir(r'C:\data\20230501\n0034')
    nodes, elems = read_inp(r'C:\data\20230501\n0034\hexmesh.inp')
    nodes = np.ascontiguousarray(nodes)
    elems = np.ascontiguousarray(elems)
    node_grid, ind_upper, ind_lower = calculate_grid(nodes, elems, calculate_lip_index=True)

    


    # regular grid by mean lip
    N = nodes
    lip_mean = N[ind_upper,:]/2 + N[ind_lower,:]/2
    N[ind_upper,:] = lip_mean
    N[ind_lower,:] = lip_mean
    N, E, ind = remove_duplicate_nodes(N, elems, stable_order=True, return_index=True, return_inverse=False)
    NG = node_grid[ind,:]
    g3d = grid_3d_from_flat(NG)
    NG_recover = grid_flat_from_3d(g3d)
    print(np.all(NG_recover == NG))
    elems_recover = elements_from_grid_3d(g3d, seed=None)
    print(np.all(elems_recover[np.argsort(elems_recover[:,0])] == E[np.argsort(E[:,0])]))

    # expand grid and extrapolate mesh
    X,Y,Z = N[:,0], N[:,1], N[:,2]
    GU,GV,GW = np.mgrid[
        -2:NG[:,0].max()+3,
        -2:NG[:,1].max()+3,
        -2:NG[:,2].max()+3,
    ]
    ind_known = bt(GU,0,g3d.shape[0]) & bt(GV,0,g3d.shape[1]) & bt(GW,0,g3d.shape[2]) # this changes as extrapolation goes on
    # g = np.concatenate((GU,GV,GW)).reshape(3,-1).T.astype(float)

    ind_find = []
    while 
    # g_keep = g3d[:,:3,:]
    # ind_find = g[:,1]<0
    # g[ind_find,:] = RBFInterpolator(NG[g_keep.flat,:], N[g_keep.flat,:], degree=3)(g[ind_find,:])
    # print('yes')

    g_keep = g3d[:5,:,:]
    ind_find = g[:,0]<0
    g[ind_find,:] = RBFInterpolator(NG[g_keep.flat,:], N[g_keep.flat,:], degree=3)(g[ind_find,:])

    g_keep = g3d[-5:, :, :]
    ind_find = g[:, 0] > NG[:, 0].max()
    g[ind_find, :] = RBFInterpolator(
        NG[g_keep.flat, :], N[g_keep.flat, :], degree=3)(g[ind_find, :])

    bone_nodes = np.nonzero(node_grid[:,1]==node_grid[:,1].max())
    faces = elems[np.isin(elems, bone_nodes).any(axis=1)]
    faces = np.ascontiguousarray(faces[:,[0,3,7,4]], dtype=np.int64)
