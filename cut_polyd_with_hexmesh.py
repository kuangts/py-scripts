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
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance, vtkClipPolyData, vtkConnectivityFilter
import shutil

def clip_polydata_with_mesh(image_polydata, mesh_polydata):

    dist = vtkImplicitPolyDataDistance()
    dist.SetInput(mesh_polydata)

    clipper = vtkClipPolyData()
    clipper.SetClipFunction(dist)
    clipper.SetInputData(image_polydata)
    clipper.InsideOutOn()
    clipper.SetValue(0.0)
    clipper.GenerateClippedOutputOff()

    largest_region = vtkConnectivityFilter()
    largest_region.SetInputConnection(clipper.GetOutputPort())
    largest_region.SetExtractionModeToLargestRegion()
    largest_region.Update()
    clipped = largest_region.GetOutput()

    return clipped




def task_06222023(ncase):
    
    os.makedirs(rf'C:\Users\tmhtxk25\Box\clipped_with_mesh\{ncase}', exist_ok=True)
    nifti_pre = glob.glob(rf'C:\data\pre-post-paired-40-send-1122\{ncase}\*-pre.nii.gz')[0]
    nifti_post = glob.glob(rf'C:\data\pre-post-paired-40-send-1122\{ncase}\*-post.nii.gz')[0]
    t_post = glob.glob(rf'C:\data\pre-post-paired-40-send-1122\{ncase}\{ncase}.tfm')[0]
    inp_mesh = rf'C:\data\meshes\{ncase}\hexmesh_open.inp'
    output = lambda f:rf'C:\Users\tmhtxk25\Box\clipped_with_mesh\{ncase}\{f}'
    ind_mat = scipy.io.loadmat(rf'C:\data\meshes\{ncase}\lip_node_index.mat')
    ind_upper = ind_mat['upper_lip_node_index']-1
    ind_lower = ind_mat['lower_lip_node_index']-1

    # nifti image -> polydata, waiting to be clipped
    img = imagedata_from_nifti(nifti_pre)
    mask = threshold_imagedata(img, foreground_threshold)
    polyd_pre = polydata_from_mask(mask)

    img = imagedata_from_nifti(nifti_post)
    mask = threshold_imagedata(img, foreground_threshold)
    polyd_post = polydata_from_mask(mask)
    polyd_post = transform_polydata(polyd_post, np.genfromtxt(t_post))
    
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
    polyd_clipped = clip_polydata_with_mesh(polyd_pre, clipper_large)
    write_polydata_to_stl(polyd_clipped, output('pre_soft_tissue_ct.stl'))
    
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

    # display
    w = Window()
    actor = w.add_polydata(clipper_large)
    actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('IndianRed'))
    actor = w.add_polydata(clipper_small)
    actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('Aqua'))
    w.start()




if __name__ == '__main__':
    task_06222023('n0056', False)
    # for f in glob.glob(rf'C:\data\meshes\n*'):
    #     try:
    #         task_06222023(os.path.basename(f))
    #     except:
    #         print(f)