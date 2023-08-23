import abc
import os, glob, sys
import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import *
from tools.image import *
from tools.polydata import *
from tools.ui import *
from tools.mesh import *
import vtk
from vtk_bridge import *
from vtkmodules.vtkFiltersGeneral import vtkClipDataSet
import shutil


ncase = 'n0042'
w = Window()

skin_pre = rf'C:\data\meshes\{ncase}\pre_skin.stl'
skin_post = rf'C:\data\meshes\{ncase}\post_skin.stl'
inp_mesh = rf'C:\data\meshes\{ncase}\hexmesh_open.inp'
output = lambda f:rf'C:\data\meshes\{ncase}\output\{f}'
ind_mat = scipy.io.loadmat(rf'C:\data\meshes\{ncase}\lip_node_index.mat')
ind_upper = ind_mat['upper_lip_node_index']-1
ind_lower = ind_mat['lower_lip_node_index']-1

polyd_pre = polydata_from_stl(skin_pre)
polyd_post = polydata_from_stl(skin_post)

# inp mesh -> polydata, extrapolate and set to be clip function
nodes, elems = read_inp(inp_mesh)
# mean lip to get regular grid
node_grid = calculate_grid(nodes, elems, calculate_lip_index=False)
lip_mean = nodes[ind_upper,:]/2 + nodes[ind_lower,:]/2
N = nodes
N[ind_upper,:], N[ind_lower,:] = lip_mean, lip_mean
N, E, ind = remove_duplicate_nodes(N, elems, stable_order=True, return_index=True, return_inverse=False)
node_grid = node_grid[ind]
F = boundary_faces(E).astype(np.int64)
mesh_surf = vtkPolyData()
mesh_surf.SetPoints(numpy_to_vtkpoints_(N))
mesh_surf.SetPolys(numpy_to_vtkpolys_(F))
write_polydata_to_stl(mesh_surf, output('mesh_surf.stl'))


# large skin - pre_soft_tissue_ct & post_skin_mesh_ct
N_, E_ = extrapolate_mesh(N, E, ((2,2),(6,2),(2,2)))
G_ = calculate_grid(N_, E_)
F_ = boundary_faces(E_, dims=((True,True),(False,True),(True,True))).astype(np.int64)
clipper_large = vtkPolyData()
clipper_large.SetPoints(numpy_to_vtkpoints_(N_))
clipper_large.SetPolys(numpy_to_vtkpolys_(F_))
write_polydata_to_stl(clipper_large, output('clipper_large.stl'))
polyd_clipped = clip_polydata_with_mesh(polyd_post, clipper_large)
write_polydata_to_stl(polyd_clipped, output('post_skin_mesh_ct.stl'))
a = w.add_polydata(polyd_clipped)
a.GetProperty().SetColor(0,1,0)
a.GetProperty().EdgeVisibilityOff()
polyd_clipped = clip_polydata_with_mesh(polyd_pre, clipper_large)
write_polydata_to_stl(polyd_clipped, output('pre_soft_tissue_ct.stl'))
a = w.add_polydata(polyd_clipped)
a.GetProperty().SetColor(1,0,0)
a.GetProperty().EdgeVisibilityOff()

# small skin - pre_skin_mesh_ct
G_clip = grid_3d_from_flat(G_)[3:-3,:,3:-3]
E_clip = elements_from_grid_3d(G_clip)
F_ = boundary_faces(E_clip, dims=((True,True),(False,True),(True,True))).astype(np.int64)
clipper_small = vtkPolyData()
clipper_small.SetPoints(numpy_to_vtkpoints_(N_))
clipper_small.SetPolys(numpy_to_vtkpolys_(F_))
write_polydata_to_stl(clipper_small, output('clipper_small.stl'))
polyd_clipped = clip_polydata_with_mesh(polyd_pre, clipper_small)
write_polydata_to_stl(polyd_clipped, output('pre_skin_mesh_ct.stl'))
a = w.add_polydata(polyd_clipped)
a.GetProperty().SetColor(0,0,1)
a.GetProperty().EdgeVisibilityOff()

# display
w.start()




