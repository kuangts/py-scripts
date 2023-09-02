import os, re, csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkExplicitStructuredGrid, vtkCellArray, VTK_HEXAHEDRON
from .ui import PolygonalSurfaceNodeSelector
from .polydata import *


def hex_from_numpy(nodes, elems, node_grid):
    N = vtkPoints()
    E = vtkCellArray()
    for n in nodes:
        N.InsertNextPoint(*n)
    for e in elems:
        E.InsertNextCell(8, e.tolist())
    H = vtkExplicitStructuredGrid()
    H.SetDimensions(node_grid.max(axis=0)+1)
    H.SetPoints(N)
    H.SetCells(E)
    H.ComputeFacesConnectivityFlagsArray()

    return H


def read_inp(file):
    with open(file,'r') as f:
        s = f.read()
    match = re.search(r'\*.*NODE[\S ]*\s+(.*)\*.*END NODE.*\*.*ELEMENT[\S ]*\s+(.*)\*.*END ELEMENT', s, re.MULTILINE | re.DOTALL)
    nodes, elems = match.group(1), match.group(2)

    # print(nodes,elems)

    nodes = np.array(list(csv.reader(nodes.strip().split('\n'))), dtype=float)[:,1:]
    elems = np.array(list(csv.reader(elems.strip().split('\n'))), dtype=int)[:,1:] - 1

    nodes, elems = np.ascontiguousarray(nodes), np.ascontiguousarray(elems)
    return nodes, elems


def write_inp(f, nodes, elems):

    ndigit = int(np.ceil(np.log10(nodes.shape[0])))
    edigit = int(np.ceil(np.log10(elems.shape[0])))

    f.write('*HEADING\r\n')
    f.write('** MESH NODES\r\n** ')
    f.write(f'Number of nodes {nodes.shape[0]}\r\n')
    f.write('*NODE\r\n')
    f.write('\n'.join(f'{i+1:>{ndigit}}, '+', '.join( f'{xx:+.6e}' for xx in x) for i,x in enumerate(nodes)) + '\n')
    f.write('*END NODE\r\n')
    
    f.write('** MESH ELEMENTS\r\n** ')
    f.write(f'Total number of elements {elems.shape[0]}\r\n')
    f.write('*ELEMENT, TYPE=C3D8\r\n')
    f.write('\n'.join(f'{i+1:>{edigit}}, '+', '.join( f'{xx:>{ndigit}}' for xx in x) for i,x in enumerate(elems+1)) + '\n')
    f.write('*END ELEMENT\r\n')
    f.write('** End of Data')
    return None


def read_feb(file):
    with open(file,'r') as f:
        Nodes, Elements = re.search(r"<Nodes[\S ]*>\s*(<.*>)\s*</Nodes>.*<Elements[\S ]*\"hex8\"[\S ]*>\s*(<.*>)\s*</Elements>", f.read(), flags=re.MULTILINE|re.DOTALL).groups()
        nodes = re.findall(r'<node id[= ]*"(\d+)">(' + ','.join([r" *[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)+ *"]*3) + r')</node>', Nodes, flags=re.MULTILINE|re.DOTALL)
        elems = re.findall(r'<elem id[= ]*"(\d+)">(' + ','.join([r' *\d+ *']*8) + r')</elem>', Elements, flags=re.MULTILINE|re.DOTALL)

    nodeid = np.asarray([int(nd[0]) for nd in nodes])-1
    nodes_with_id = np.asarray([nd[1].split(',') for nd in nodes], dtype=float)
    elems = np.asarray([el[1].split(',') for el in elems], dtype=int)-1
    nodes = np.empty((nodeid.max()+1,3))
    nodes[:] = np.nan
    nodes[nodeid,:] = nodes_with_id[:]
    return nodes, elems


def default_grid_seed():
    return np.array([
        [1,1,0,0,1,1,0,0],
        [1,0,0,1,1,0,0,1],
        [1,1,1,1,0,0,0,0]]).T



def calculate_grid(nodes, elems, seed=None, calculate_lip_index=False):
    if not seed:
        seed = default_grid_seed()
    seed = seed - seed.min(axis=0)
    node_grid = -np.empty_like(nodes)
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
        node_grid[sub_ele[(sub_r, sub_c)],:] = node_grid[sub_ele[(sub_r, c[sub_r])],:] + seed[sub_c,:] - seed[c[sub_r],:]
    
    assert not np.isnan(node_grid).any(), 'error calculate grid'

    node_grid = (node_grid - node_grid.min(axis=0)).astype(elems.dtype)
    # elem_grid = node_grid[elems.T,:].mean(axis=0)

    if calculate_lip_index:
        g, ind = np.unique(node_grid, axis=0, return_inverse=True)
        ind_duplicate_grid_positions = np.nonzero(np.bincount(ind)!=1)[0]
        assert len(np.unique(g[ind_duplicate_grid_positions,2]))==1, 'check mesh'
        g_dup = g[ind_duplicate_grid_positions,:]

        I = np.concatenate(np.mgrid[
            g_dup[:,0].min():g_dup[:,0].max()+1,
            g_dup[:,1].min():g_dup[:,1].max()+1,
            g_dup[:,2].min():g_dup[:,2].max()+1,
        ]).reshape(3,-1).T

        nbrs = NearestNeighbors(n_neighbors=2).fit(node_grid)
        D, ind_lip = nbrs.kneighbors(I)
        elems_lip = elems[np.isin(elems, ind_lip).any(axis=1),:]
        elems_lip_center = node_grid[elems_lip.T,:].mean(axis=0)
        lip_w = elems_lip_center[:,2].mean()
        ind_upper, ind_lower = \
            ind_lip[np.isin(ind_lip, elems_lip[elems_lip_center[:,2]>lip_w])].reshape(-1,6),\
            ind_lip[np.isin(ind_lip, elems_lip[elems_lip_center[:,2]<lip_w])].reshape(-1,6)
        # return node_grid, ind_upper, ind_lower
        return node_grid, ind_upper[::-1], ind_lower[::-1] # reorder for legacy reasons
    else:
        return node_grid


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


def grid_3d_from_flat(node_grid):
    g3d = -np.ones(node_grid.max(axis=0)+1, dtype=int)
    g3d[(*node_grid.T,)] = np.arange(node_grid.shape[0])
    return g3d


def grid_flat_from_3d(g3d):
    node_grid = np.empty((g3d.max()+1,3))
    node_grid[:] = np.nan
    node_grid[g3d.flat,:] = np.concatenate(np.mgrid[0:g3d.shape[0], 0:g3d.shape[1], 0:g3d.shape[2]]).reshape(3,-1).T
    return node_grid


def elements_from_grid_3d(g3d, seed=None):
    if not seed:
        seed = default_grid_seed()

    I = np.ogrid[:g3d.shape[0]-1,:g3d.shape[1]-1,:g3d.shape[2]-1]
    elems = np.array([g3d[(I[0]+s[0],I[1]+s[1],I[2]+s[2])] for s in seed]).T.reshape(-1,8).astype(np.int64)
    return elems


def boundary_faces(elems, seed=None, dims=((True,)*2,)*3):
    if not seed:
        seed = default_grid_seed()
    nbrs = NearestNeighbors(n_neighbors=1).fit(seed)
    ele_ind = lambda g: nbrs.kneighbors(g)[1].flatten()
    fg = np.array(((0,0),(1,0),(1,1),(0,1)))
    faces = np.empty((0,4), dtype=int)
    for d,vv in enumerate(dims):
        for v in np.nonzero(vv)[0]:
            fg3 = ele_ind(np.insert(fg,d,v,axis=1))
            fg3 = fg3[::(1 if v else -1) * (-1 if d==1 else 1)] # this makes sure the normal direction is correct
            fg3_ = ele_ind(np.insert(fg,d,1-v,axis=1))
            node_id = np.setdiff1d(elems[:,fg3],elems[:,fg3_])
            f = elems[np.isin(elems[:,fg3], node_id).any(axis=1)][:,fg3]
            faces = np.vstack((faces,f))
    return faces


def local_matrices(N,E):
    gc = [
        [1,1,0,0,1,1,0,0],
        [1,0,0,1,1,0,0,1],
        [1,1,1,1,0,0,0,0],
    ]
    gc = np.array(gc).T
    ind = np.empty(gc.shape, dtype=int)
    ind[...] = 0
    for i in range(E.shape[1]):
        edg = gc-gc[i]
        ind[i] = np.nonzero(np.sum(edg**2,1)**.5==1)[0]
        jac = edg[ind[i],:].T
        vol = np.linalg.det(jac)
        if vol < 0:
            ind[i] = ind[i,::-1]
    
    return N[E[:,ind]] - N[E[...,None]]


def local_volume(N,E):

    edg = local_matrices(N,E)
    vol = np.linalg.det(edg)
    return vol


def jacobian_ratio(N,E):
    edg = local_matrices(N,E)
    vol = np.linalg.det(edg)

    # setting the negative volumes to zero to allow calculation to procede
    print((vol<=0).sum())
    vol[vol<0] = 0

    dia = np.sum(edg**2, axis=(-2,-1))
    val = 24./np.sum(dia/vol**(2/3), axis=1)

    return val


def extrapolate_mesh(nodes, elems, bd_size):
    def bt(a:np.ndarray, lo, hi):
        return np.logical_and(a>=lo, a<hi)

    N, E = nodes, elems
    NG = calculate_grid(nodes, elems)
    g3d = grid_3d_from_flat(NG)
    GU,GV,GW = np.mgrid[
        -bd_size[0][0]:NG[:,0].max()+bd_size[0][1]+1,
        -bd_size[1][0]:NG[:,1].max()+bd_size[1][1]+1,
        -bd_size[2][0]:NG[:,2].max()+bd_size[2][1]+1,
    ]
    ind_known = bt(GU,0,g3d.shape[0]) & bt(GV,0,g3d.shape[1]) & bt(GW,0,g3d.shape[2])
    g_ = np.concatenate((GU,GV,GW)).reshape(3,-1).T.astype(float)
    g3d_ = np.arange(GU.size).reshape(GU.shape)
    N_ = np.empty(g_.shape)
    N_[g3d_[ind_known]] = N[g3d.flat,:]
    NB = (3,1,3)
    BU, BV, BW = np.meshgrid(
        np.round(np.arange(NB[0]+1)*ind_known.shape[0]/NB[0]).astype(int),
        np.round(np.arange(NB[1]+1)*ind_known.shape[1]/NB[1]).astype(int),
        np.round(np.arange(NB[2]+1)*ind_known.shape[2]/NB[2]).astype(int), 
        indexing='ij')

    for i,j,k in np.ndindex(NB):
        ind_block = np.zeros(g3d_.shape, dtype=bool)
        ind_block[
            BU[i,j,k]:BU[i+1,j,k],
            BV[i,j,k]:BV[i,j+1,k],
            BW[i,j,k]:BW[i,j,k+1],
            ] = True
        
        ind_take = g3d_[np.logical_and(binary_dilation(ind_block, structure=np.ones((10,10,10))), ind_known)]
        ind_find = g3d_[np.logical_and(ind_block, ~ind_known)]

        N_[ind_find,:] = RBFInterpolator(g_[ind_take,:], N_[ind_take,:], degree=3)(g_[ind_find,:])
    E_ = elements_from_grid_3d(g3d_)
    return N_, E_


class HexahedralMeshFix(PolygonalSurfaceNodeSelector):
    '''use this class to fix minor hexahedral mesh defects
        ```
        sel = HexahedralMeshFix()
        sel.initialize(r'C:\data\meshes\n0030\test\hexmesh_open_test.inp')
        sel.start()

        ```
    '''
    def initialize(self, inp_file, override=False):

        self.file_path = inp_file
        self.override = override
        nodes, elems = read_inp(inp_file)
        self.nodes, self.elems = nodes, elems 
        self.node_grid = calculate_grid(nodes, elems)

        self.points = vtkPoints()
        for node in nodes.tolist():
            self.points.InsertNextPoint(node)

        faces = boundary_faces(elems, dims=((False, False), (True, True), (False, False)))
        other_faces = boundary_faces(elems, dims=((True, True), (False, False), (True, True)))

        pick_surf = vtkPolyData()
        pick_surf.SetPoints(self.points)
        E = vtkCellArray()
        for f in faces.tolist():
            E.InsertNextCell(len(f),f)
        pick_surf.SetPolys(E)

        show_surf = vtkPolyData()
        show_surf.SetPoints(self.points)
        E = vtkCellArray()
        for f in other_faces.tolist():
            E.InsertNextCell(len(f),f)
        show_surf.SetPolys(E)

        self.pick_surf, self.show_surf = pick_surf, show_surf

        return super().initialize(self.pick_surf, self.show_surf)


    def _move(self):

        nodes = vtk_to_numpy(self.points.GetData()) # shares memory with vtk
        elems = self.elems
        node_grid = self.node_grid

        ind_node_select = vtk_to_numpy(self.selection).flatten()==1
        ind_node_select = ind_node_select.nonzero()[0]
        ind_node_change_3d = np.all(node_grid[:,None,:] == node_grid[ind_node_select,:][None,:,:], axis=2)
        d1,d2 = np.nonzero(ind_node_change_3d)
        ind_node_change = d1[np.argsort(d2)]

        ind_node_faces = np.nonzero(np.logical_or(node_grid[:,1]==node_grid[:,1].min(), node_grid[:,1]==node_grid[:,1].max()))[0]
        ind_node_change_neighbor = ind_node_change.copy()
        ind_node_change_neighbor = np.all(node_grid[:,None,(0,2)] == node_grid[ind_node_change,:][None,:,(0,2)], axis=2)
        ind_node_change_neighbor = np.nonzero(ind_node_change_neighbor.any(axis=1))[0]
        for _ in range(2):
            ind_node_change_neighbor = elems[np.isin(elems,ind_node_change_neighbor).any(axis=1),:].flatten()

        ind_node_change_neighbor = np.unique(ind_node_change_neighbor)
        ind_node_change_neighbor_interp = ind_node_change_neighbor.copy()
        for _ in range(3):
            ind_node_change_neighbor_interp = elems[np.isin(elems,ind_node_change_neighbor_interp).any(axis=1),:].flatten()

        ind_node_change_neighbor_interp = np.unique(ind_node_change_neighbor_interp)
        ind_node_change = np.union1d(ind_node_change, np.setdiff1d(ind_node_change_neighbor, ind_node_faces))
        ind_node_interp = np.setdiff1d(ind_node_change_neighbor_interp, ind_node_change)

        # nodes are modified here
        nodes[ind_node_change,:] = RBFInterpolator(node_grid[ind_node_interp,:], nodes[ind_node_interp,:], degree=3)(node_grid[ind_node_change,:])

        self.points.Modified()
        self._deselect()
        self.render_window.Render()

        return None


    def save(self):
        if self.override:
            with open(self.file_path, 'w', newline='') as f:
                write_inp(f, vtk_to_numpy(self.points.GetData()), self.elems)
        else:
            self.save_ui(title='Save landmarks to *.csv ...',
                         initialdir=os.path.dirname(self.file_path),
                         initialfile=os.path.basename(self.file_path))
        
        return None





# def seed_grid(nodes, elems):
#     N, E = nodes, elems
#     ng = np.zeros_like(N)
#     # assumes u,v,w correspond to x,y,z, in both dimension and direction
#     # first, find 8 nodes with single occurence
#     occr = np.bincount(E.flat)
#     n8 = np.where(occr==1)[0]
#     cols = []
#     for n in n8:
#         cols.append(np.mgrid[range(E.shape[0]), range(E.shape[1])][1][np.isin(E,n)][0])
#     n8[cols] = n8

#     assert n8.size==8, 'check mesh'
#     # then, set the grid position of these eight nodes
#     # set left and right (-x -> -INF, +x -> +INF)
#     ng[n8,0] = np.where(N[n8,0]<np.median(N[n8,0]), np.NINF, np.PINF)
#     # set up and down (-z -> -INF, +z -> +INF)
#     ng[n8,2] = np.where(N[n8,2]<np.median(N[n8,2]), np.NINF, np.PINF)
#     # set front and back
#     n4 = n8[N[n8,2]<np.median(N[n8,2])] # top 4
#     c = N[n4].mean(axis=0)
#     n2 = n4[N[n4,0]<np.median(N[n4,0])] # top left
#     d = np.sum((N[n2] - c)**2, axis=1)**.5
#     ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
#     n2 = n4[N[n4,0]>np.median(N[n4,0])] # top right
#     d = np.sum((N[n2] - c)**2, axis=1)**.5
#     ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

#     n4 = n8[N[n8,2]>np.median(N[n8,2])] # bottom 4
#     c = N[n4].mean(axis=0)
#     n2 = n4[N[n4,0]<np.median(N[n4,0])] # bottom left
#     d = np.sum((N[n2] - c)**2, axis=1)**.5
#     ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
#     n2 = n4[N[n4,0]>np.median(N[n4,0])] # bottom right
#     d = np.sum((N[n2] - c)**2, axis=1)**.5
#     ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

#     seed = np.ones((8,3))*.5
#     ind_preset = np.all(np.isinf(ng), axis=1).nonzero()[0]
#     for row,col in zip(*np.where(np.isin(N, ind_preset))):
#         seed[col] *= np.sign(ng[N[row,col]])
#     return seed



if __name__=='__main__':
    
    nodes, elems = read_inp(r'C:\data\20230501\n0034\hexmesh.inp')
    node_grid, ind_upper, ind_lower = calculate_grid(nodes, elems, calculate_lip_index=True)

    ff = boundary_faces(elems)


    import scipy
    ind_mat = scipy.io.loadmat(r'C:\data\20230501\n0034\lip_node_index.mat')
    assert np.all(ind_mat['upper_lip_node_index']-1 == ind_upper) and np.all(ind_mat['lower_lip_node_index']-1 == ind_lower), 'not the same as before'

    lip_mean = nodes[ind_upper,:]/2 + nodes[ind_lower,:]/2
    nodes[ind_upper,:] = lip_mean
    nodes[ind_lower,:] = lip_mean










    pass