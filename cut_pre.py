
import os, glob, sys
import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import *
from tools import *
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

def load_landmark(file):
    with open(file, 'r') as f:
        axyz = list(csv.reader(f))
    return {b[0]:[float(a) for a in b[1:]] for b in axyz}



case_name = 'n0060'

nodes, elems = read_inp(rf'C:\Users\tmhtxk25\Box\Facial Prediction_DK_TK\recent_cases\{case_name}\hexmesh_open.inp')
ind_mat = scipy.io.loadmat(rf'C:\Users\tmhtxk25\Box\Facial Prediction_DK_TK\recent_cases\{case_name}\lip_node_index.mat')
polyd_pre = polydata_from_stl(rf'C:\Users\tmhtxk25\Box\clipped_with_mesh\{case_name}\pre_soft_tissue_ct.stl')
polyd_out = rf'C:\Users\tmhtxk25\Box\clipped_with_mesh\{case_name}\pre_skin_mesh_ct_try.stl'
lmk_pre_maxi = np.genfromtxt(os.path.join(r'C:\data\pre-post-paired-40-send-1122', case_name, 'maxilla_landmark.txt'))
lmk_pre_skin = np.genfromtxt(os.path.join(r'C:\data\pre-post-paired-40-send-1122', case_name, 'skin_landmark.txt'))
lmk_pre = load_landmark(os.path.join(r'C:\data\pre-post-paired-soft-tissue-lmk-23', case_name, 'skin-pre-23.csv'))
if np.isnan(lmk_pre_maxi).any():
    lmk_pre_maxi = np.genfromtxt(os.path.join(r'C:\data\pre-post-paired-40-send-1122', case_name, 'maxilla_landmark.txt'), delimiter=',')
if np.isnan(lmk_pre_skin).any():
    lmk_pre_skin = np.genfromtxt(os.path.join(r'C:\data\pre-post-paired-40-send-1122', case_name, 'skin_landmark.txt'), delimiter=',')
ind_upper = ind_mat['upper_lip_node_index']-1
ind_lower = ind_mat['lower_lip_node_index']-1

node_grid = calculate_grid(nodes, elems, calculate_lip_index=False)
lip_mean = nodes[ind_upper,:]/2 + nodes[ind_lower,:]/2
N = nodes
N[ind_upper,:], N[ind_lower,:] = lip_mean, lip_mean
N, E, ind = remove_duplicate_nodes(N, elems, stable_order=True, return_index=True, return_inverse=False)
node_grid = node_grid[ind]

# small skin - pre_skin_mesh_ct
G_clip = grid_3d_from_flat(node_grid)[1:-1,:,1:-1]
E_clip = elements_from_grid_3d(G_clip)
F_ = boundary_faces(E_clip, dims=((True,True),(False,True),(True,True))).astype(np.int64)
clipper_small = vtkPolyData()
clipper_small.SetPoints(numpy_to_vtkpoints_(nodes))
clipper_small.SetPolys(numpy_to_vtkpolys_(F_))
polyd_clipped = clip_polydata_with_mesh(polyd_pre, clipper_small)
write_polydata_to_stl(polyd_clipped, polyd_out)

ff_horizontal_normal = np.cross(lmk_pre_maxi[15] - lmk_pre_maxi[9], lmk_pre_maxi[16] - lmk_pre_maxi[10])
polyd_clipped = cut_polydata(polyd_clipped, np.array(lmk_pre['Prn']) + [0,0,10], ff_horizontal_normal)
write_polydata_to_stl(polyd_clipped, polyd_out)
