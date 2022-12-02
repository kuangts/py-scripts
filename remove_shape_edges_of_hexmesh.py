import os, sys, re

import numpy as np
import open3d as o3d
import scipy as sp


def seed_grid(nodes, elems):
    N, E = nodes, elems
    seed = np.zeros((8,3))
    # assumes u,v,w correspond to x,y,z, in both dimension and direction
    # first, find 8 nodes with single occurence
    n8 = N[np.where(np.isin(elems, np.where(np.bincount(E.flat)==1)[0]), elems, 0).sum(axis=0),:]
    # then, set the grid position of these eight nodes
    ## set left and right (-x -> -INF, +x -> +INF)
    ## set up and down (-z -> -INF, +z -> +INF)
    seed[:,0] = np.where(n8[:,0]<np.median(n8[:,0]), -1, 1)
    seed[:,2] = np.where(n8[:,2]<np.median(n8[:,2]), -1, 1)
    # set front and back
    i4 = np.where(n8[:,2]<np.median(n8[:,2]))[0] # top 4
    seed[i4[np.argsort(n8[i4,0])],1] = [1, -1, -1, 1]
    i4 = np.where(n8[:,2]>np.median(n8[:,2]))[0] # bottom 4
    seed[i4[np.argsort(n8[i4,0])],1] = [1, -1, -1, 1]

    return seed



def calculate_node_grid(elems, seed=None):
    ng = np.empty((elems.max()+1,3))
    ng[:] = np.nan

    seed = np.array(seed, dtype=float)
    seed -= np.min(seed, axis=0)
    seed /= (np.max(seed, axis=0)-np.min(seed, axis=0))
    seed -= np.mean(seed, axis=0)
    # calculate grid by expending from seed
    newly_set = np.array([0], dtype=int)
    ng[newly_set,:] = 0
    ind_unset = np.any(np.isnan(ng), axis=1)
    while np.any(ind_unset):
        elem_set = np.isin(elems, newly_set)
        row_num = np.any(elem_set, axis=1)
        elem, elem_set = elems[row_num], elem_set[row_num]
        for row in range(elem_set.shape[0]):
            col = elem_set[row].nonzero()[0][0]
            ng[elem[row],:] = seed + (ng[elem[row,col],:] - seed[col,:])
        newly_set = np.intersect1d(elem, np.where(ind_unset)[0])
        ind_unset[newly_set] = False

    ng -= np.min(ng, axis=0)
    ng = ng.round().astype(int)
    return ng


def quadrilateral_face(elems, node_grid, node_index=None, element_index=None):
    if not elems.size:
        return np.empty((0,4))
    g = node_grid[elems[0,:],:]
    g = g-g.mean(axis=0)
    column_index = [d.nonzero()[0] for d in (g<0).T] + [d.nonzero()[0] for d in (g>0).T]
    for i,c in enumerate(column_index):
        gc = np.delete(g[c], i%3, axis=1)
        c = c[np.argsort(np.sign(gc[:,0])+(gc[:,0] != gc[:,1]))]
        column_index[i] = c
        sign = np.dot(np.cross( g[c[1]]-g[c[0]], g[c[-1]]-g[c[0]], axis=0), g[c].mean(axis=0)) > 0
        if not sign:
            column_index[i] = c[::-1]
            
    if element_index is not None:
        elems = elems[element_index,:]
    f = np.vstack([ elems[:,ind] for ind in column_index ])
    _, fid = np.unique([np.roll(ff, -np.argmin(ff)) for ff in f], axis=0, return_index=True) # find unique faces up to a cyclic permutation
    f = f[fid,:]

    if node_index is not None:
        if np.all(np.isin(node_index, [True,False])):
            node_index = node_index.nonzero()[0]
        f = f[np.all(np.isin(f, node_index), axis=1),:]
    return f



def main(file):

    # load mandible stl
    mand = o3d.io.read_triangle_mesh(os.path.join(os.path.dirname(file),'mandible_surface.stl'))
    mand.compute_vertex_normals()

    # parse .inp file
    with open(file,'r') as f:
        match = re.search(
            r'\*.*NODE[\S ]*\s+(.*)\*.*END NODE.*\*.*ELEMENT[\S ]*\s+(.*)\*.*END ELEMENT', 
            f.read(), 
            re.MULTILINE | re.DOTALL)
        try:
            nodes, elems = match.group(1), match.group(2)
        except Exception as e:
            print(e)
            raise ValueError('the file cannot be read for nodes and elements')
    nodes4 = [node.split(',') for node in nodes.strip().split('\n')]
    nodes = np.asarray(nodes4, dtype=float)[:,1:]
    elems9 = [elem.split(',') for elem in elems.strip().split('\n')]
    elems = np.asarray(elems9, dtype=int)[:,1:]-1

    # calculate grid
    seed = seed_grid(nodes, elems)
    node_grid = calculate_node_grid(elems, seed)
    problem_node_id = np.isin(node_grid[:,2], np.arange(3)) & np.isin(node_grid[:,1], np.arange(3))

    # show all the visible faces
    faces = quadrilateral_face(elems, node_grid)
    _, ind = np.unique(faces, axis=0, return_index=True)
    faces = faces[np.bincount(ind)==1,:]

    # for visualization

    # fix problem nodes
    elems_interp = elems[np.any(np.isin(elems, np.where(problem_node_id)[0]), axis=1),:]
    for _ in range(10):
        elems_interp = elems[np.any(np.isin(elems, elems_interp.flat), axis=1),:]
    node_id_interp = np.setdiff1d(np.unique(elems_interp), np.where(problem_node_id)[0])
    rbf = sp.interpolate.RBFInterpolator(node_grid[node_id_interp,:], nodes[node_id_interp,:])
    nodes_new = np.copy(nodes)
    nodes_new[problem_node_id,:] = np.nan
    nodes_new[problem_node_id,:] = rbf(node_grid[problem_node_id,:])

    # visualize
    problem_faces = faces[np.any(np.isin(faces, np.where(problem_node_id)[0]), axis=1),:]
    problem_edges = o3d.geometry.LineSet.create_from_triangle_mesh(
        o3d.geometry.TriangleMesh(
            vertices = o3d.utility.Vector3dVector(nodes),
            triangles = o3d.utility.Vector3iVector(np.vstack((problem_faces[:,[0,1,2]],problem_faces[:,[0,2,3]])).astype(int)),
        )
    ).paint_uniform_color([.5,.2,.2])
    s = o3d.geometry.TriangleMesh(
        vertices = o3d.utility.Vector3dVector(nodes_new),
        triangles = o3d.utility.Vector3iVector(np.vstack((faces[:,[0,1,2]],faces[:,[0,2,3]])).astype(int)),
    ).paint_uniform_color([.9,.7,.5])
    vc = np.asarray(s.vertex_colors)
    vc[node_id_interp,0] = .5
    vc[problem_node_id,:] = [.8,.8,0]
    s.vertex_colors = o3d.utility.Vector3dVector(vc)
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(s).paint_uniform_color([.2,.5,.2])
    o3d.visualization.draw_geometries([s, problem_edges, edges], mesh_show_back_face=True)

    with open(os.path.splitext(file)[0]+'_new_edge.inp','w',newline='') as f:

        ndigit = int(np.ceil(np.log10(nodes_new.shape[0])))
        edigit = int(np.ceil(np.log10(elems.shape[0])))

        f.write('*HEADING\r\n')
        f.write('** MESH NODES\r\n** ')
        f.write(f'Number of nodes {nodes_new.shape[0]}\r\n')
        f.write('*NODE\r\n')
        f.write('\n'.join(f'{i+1:>{ndigit}}, '+', '.join( f'{xx:+.6e}' for xx in x) for i,x in enumerate(nodes_new)) + '\n')
        f.write('*END NODE\r\n')
        
        f.write('** MESH ELEMENTS\r\n** ')
        f.write(f'Total number of elements {elems.shape[0]}\r\n')
        f.write('*ELEMENT, TYPE=C3D8\r\n')
        f.write('\n'.join(f'{i+1:>{edigit}}, '+', '.join( f'{xx:>{ndigit}}' for xx in x) for i,x in enumerate(elems)) + '\n')
        f.write('*END ELEMENT\r\n')
        f.write('** End of Data')


if __name__=='__main__':
    main(r'C:\OneDrive\FEM Mesh Generation\Cases\n0007\hexmesh_open.inp')