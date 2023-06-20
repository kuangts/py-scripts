import glob, os, csv, shutil, re
from os.path import join as pjoin
from os.path import exists as pexists
from os.path import isfile as isfile
from os.path import isdir as isdir
from os.path import basename, dirname, normpath, realpath


import numpy as np
import vtk
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkTriangleFilter, vtkClipPolyData
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkImageData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkTransformFilter
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActorCollection,
    vtkTextActor,    
    vtkProperty,
    vtkCellPicker,
    vtkPointPicker,
    vtkPolyDataMapper,
    vtkDataSetMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk


from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkSmoothPolyDataFilter
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkDiscreteFlyingEdges3D, vtkTransformFilter
from vtkmodules.vtkIOImage import vtkNIFTIImageReader, vtkNIFTIImageHeader
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkCommonCore import vtkPoints

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkDataSetAttributes,
    vtkIterativeClosestPointTransform,
    vtkPlane
)
from vtkmodules.vtkFiltersCore import (
    vtkMaskFields,
    vtkThreshold,
    vtkWindowedSincPolyDataFilter
)
from vtkmodules.vtkFiltersGeneral import (
    vtkDiscreteFlyingEdges3D,
    vtkDiscreteMarchingCubes
)
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from vtkmodules.vtkImagingStatistics import vtkImageAccumulate
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkImagingMorphological import vtkImageOpenClose3D


from vtk_bridge import *

colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']

colors = vtkNamedColors()

soft_tissue_threshold = (324, 1249)
bone_threshold = (1250, 4095)
all_threshold = (324, 4095)


def load_nifti(file):
    src = vtkNIFTIImageReader()
    src.SetFileName(file)
    src.Update()
    matq = vtkmatrix4x4_to_numpy_(src.GetQFormMatrix())
    mats = vtkmatrix4x4_to_numpy_(src.GetSFormMatrix())
    assert np.allclose(matq, mats), 'nifti image qform and sform matrices not the same, requires attention'
    origin = matq[:3,:3] @ matq[:3,3]
    img = vtkImageData()
    img.DeepCopy(src.GetOutput())
    img.SetOrigin(origin.tolist())
    return img


# def write_image_to_dicom()


def threshold_image(image_data, thresh):
    threshold = vtkImageThreshold()
    threshold.SetInputData(image_data)
    threshold.SetInValue(1.0)
    threshold.SetOutValue(0.0)
    threshold.ReplaceInOn()
    threshold.ReplaceOutOn()
    threshold.ThresholdBetween(*thresh)
    threshold.Update()
    return threshold.GetOutput()


def mask_to_object(mask_image):
    # closer = vtkImageOpenClose3D()
    discrete_cubes = vtkDiscreteFlyingEdges3D()
    smoother = vtkWindowedSincPolyDataFilter()
    scalars_off = vtkMaskFields()
    geometry = vtkGeometryFilter()

    smoothing_iterations = 15
    pass_band = 0.001
    feature_angle = 120.0
    
    # closer.SetInputData(label_data)
    # closer.SetKernelSize(2, 2, 2)
    # closer.SetCloseValue(1.0)

    discrete_cubes.SetInputData(mask_image)
    discrete_cubes.SetValue(0, 1.0)
    
    scalars_off.SetInputConnection(discrete_cubes.GetOutputPort())
    scalars_off.CopyAttributeOff(vtkMaskFields().POINT_DATA,
                                 vtkDataSetAttributes().SCALARS)
    scalars_off.CopyAttributeOff(vtkMaskFields().CELL_DATA,
                                 vtkDataSetAttributes().SCALARS)

    geometry.SetInputConnection(scalars_off.GetOutputPort())

    smoother.SetInputConnection(geometry.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    
    smoother.Update()

    return smoother.GetOutput()


def write_polydata(polyd:vtkPolyData, file:str):
    writer = vtkSTLWriter()
    writer.SetInputData(polyd)
    writer.SetFileName(file)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def read_polydata(file:str):
    reader = vtkSTLReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()


def translate_polydata(polyd:vtkPolyData, t:numpy.ndarray):
    T = vtkTransform()
    T.Translate(t)
    Transform = vtkTransformPolyDataFilter()
    Transform.SetTransform(T)
    Transform.SetInputData(polyd)
    Transform.Update()
    return Transform.GetOutput()


def transform_polydata(polyd, t:numpy.ndarray):
    T = vtkTransform()
    M = vtkMatrix4x4()
    M.DeepCopy(t.ravel())
    T.SetMatrix(M)
    Transform = vtkTransformPolyDataFilter()
    Transform.SetTransform(T)
    Transform.SetInputData(polyd)
    Transform.Update()
    return Transform.GetOutput()


def flip_normal_polydata(polyd):
    T = vtkPolyDataNormals()
    T.FlipNormalsOn()
    T.SetInputData(polyd)
    T.Update()
    return T.GetOutput()


def cut_polydata(polyd, plane_origin, plane_normal):
    plane = vtkPlane()
    plane.SetNormal(*plane_normal)
    plane.SetOrigin(*plane_origin)

    cutter = vtkClipPolyData()
    cutter.SetClipFunction(plane)
    cutter.SetInputData(polyd)
    cutter.Update()

    return cutter.GetOutput()


def polydata_actor(polyd, **property):
    
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polyd)
    actor = vtkActor()
    actor.SetMapper(mapper)
    if property is not None:
        for pk,pv in property.items():
            if pk=='Color' and isinstance(pv, str):
                pv = colors.GetColor3d(pv)
            getattr(actor.GetProperty(),'Set'+pk).__call__(pv)
    return actor


def render_window(case_name):
    renderer = vtkRenderer()
    renderer.SetBackground(.67, .93, .93)

    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000,1500)
    renderWindow.SetWindowName(case_name)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    style = vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)
    return renderWindow, renderer, interactor


def show_polydata(case_name, polyds, properties=None):
    
    renderWindow, renderer, interactor = render_window(case_name)
    for d, p in zip(polyds, properties):
        renderer.AddActor(polydata_actor(d, p))

    renderWindow.Render()
    interactor.Start()


def show_actors(case_name, actors):
    
    renderWindow, renderer, interactor = render_window(case_name)
    for a in actors:
        renderer.AddActor(a)

    renderWindow.Render()
    interactor.Start()


def load_landmark(file):
    with open(file, 'r') as f:
        axyz = list(csv.reader(f))
    return {b[0]:[float(a) for a in b[1:]] for b in axyz}


def landmark_polydata(lmk):

    lmk_np_nx3 = lmk
    if isinstance(lmk, dict):
        lmk_np_nx3 = np.array(list(lmk.values()))
    lmk_np_nx3 = np.asarray(lmk_np_nx3)

    input = vtkPolyData()
    input.SetPoints(numpy_to_vtkpoints_(lmk_np_nx3))

    glyphSource = vtkSphereSource()
    glyphSource.SetRadius(1)
    glyphSource.Update()

    glyph3D = vtkGlyph3D()
    glyph3D.GeneratePointIdsOn()
    glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
    glyph3D.SetInputData(input)
    glyph3D.SetScaleModeToDataScalingOff()
    glyph3D.Update()

    return glyph3D.GetOutput()


def landmark_actors(lmk):

    actors = [
            polydata_actor(landmark_polydata(lmk), Color=colors.GetColor3d('tomato')),
        ]
    
    if isinstance(lmk, dict): # add labels
        for label, coord in lmk.items():
            txt = vtkBillboardTextActor3D()
            txt.SetPosition(*coord)
            txt.SetInput(str(label))
            txt.GetTextProperty().SetFontSize(24)
            txt.GetTextProperty().SetJustificationToCentered()
            txt.GetTextProperty().SetColor((0,0,0))
            txt.GetTextProperty().SetOpacity(.5)
            txt.ForceOpaqueOn()
            txt.SetDisplayOffset(0,10)
            txt.PickableOff()
            actors.append(txt)

    return actors


def write_landmark(lmk, file):
    with open(file, 'w', newline='') as f:
        csv.writer(f).writerows([[k,*v] for k,v in lmk.items()])


def register(source, target, result):

    T = vtkIterativeClosestPointTransform()

    reader = vtkSTLReader()
    reader.SetFileName(source)
    reader.Update()
    T.SetSource(reader.GetOutput())
    reader = vtkSTLReader()
    reader.SetFileName(target)
    reader.Update()
    T.SetTarget(reader.GetOutput())

    T.Update()
    t = vtkmatrix4x4_to_numpy_(T.GetMatrix())
    np.savetxt(result, t, '%+.8e')

    
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



def calculate_node_grid(nodes, elems, seed=seed()):
    ng = np.empty(nodes.shape)
    ng[:] = np.nan

    seed = np.array(seed, dtype=float)
    seed -= np.min(seed, axis=0)
    seed /= (np.max(seed, axis=0)-np.min(seed, axis=0))
    # seed -= np.mean(seed, axis=0)
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
            ng[elem[row],:] = seed - seed[col,:] + ng[elem[row,col],:]
        newly_set = np.intersect1d(elem, np.where(ind_unset)[0])
        ind_unset[newly_set] = False

    ng -= np.min(ng, axis=0)
    ng = ng.round().astype(int)
    return ng


def grid_3d(node_grid):
    g3d = -np.ones(node_grid.max(axis=0)+1, dtype=int)
    g3d[(*node_grid.T,)] = np.arange(node_grid.shape[0])
    # for i,ng in enumerate(node_grid):
    #     print(ng)
    #     g3d[*ng] = i
    return g3d


def glob_root(d, root_dir='.'):
    current_dir = os.getcwd()
    os.chdir(root_dir)
    f = glob.glob(d)
    os.chdir(current_dir)
    return f


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


    box_root = r'C:\Users\tians\Box'
    data_root = r'C:\Users\tians\Box\RPI\data'
    working_root = rf'P:\20230615'
    all_cases_root = rf'{data_root}\pre-post-paired-unc40-send-1122'
    lmk_root = rf'{data_root}\pre-post-paired-soft-tissue-lmk-23'
    mesh_root = rf'C:\Users\tians\OneDrive\meshes'
    list_of_files = [
        'hexmesh_open.inp', 
        'pre_skin_mesh_ct.stl', 
        'pre_soft_tissue_ct.stl', 
        'post_skin_mesh_ct.stl', 
        'movement_di.tfm', 
        'movement_diL.tfm', 
        'movement_diR.tfm', 
        'movement_le.tfm', 
        'pre_di.stl',
        'pre_diL.stl',
        'pre_diR.stl',
        'pre_le.stl',
        ]

    override = True # completely redo
    inspection = False # inspect cases
    other_task = False
    cases_working = ['n0056']
    print(f'working on {len(cases_working)} cases:', *cases_working)

    os.makedirs(working_root, exist_ok=True)

    for case_name in cases_working:

        case_dir = pjoin(working_root, case_name)
        if all([isfile(pjoin(case_dir, x)) for x in list_of_files]):
            print(f'{case_name} is completed.')
            if not override and not inspection and not other_task:
                continue
        else:
            print(f'{case_name} is not complete.')
        os.makedirs(case_dir, exist_ok=True)

        #__________________________________________________________________________________________#
        # first processing, write pre and registered post stl and lmk, cut stl

        if override or not all([isfile(pjoin(case_dir, x)) for x in ('skin-pre.stl', 'skin-post.stl', 'skin-pre-23.csv', 'skin-post-23.csv', 'pre_skin_mesh_ct.stl', 'pre_soft_tissue_ct.stl', 'post_skin_mesh_ct.stl')]):

            lmk_pre = load_landmark(pjoin(lmk_root, case_name, 'skin-pre-23.csv'))

            lmk_pre_maxi = np.genfromtxt(pjoin(all_cases_root, case_name, 'maxilla_landmark.txt'))
            lmk_pre_mand = np.genfromtxt(pjoin(all_cases_root, case_name, 'mandible_landmark.txt'))
            lmk_pre_skin = np.genfromtxt(pjoin(all_cases_root, case_name, 'skin_landmark.txt'))
            if np.isnan(lmk_pre_maxi).any():
                lmk_pre_maxi = np.genfromtxt(pjoin(all_cases_root, case_name, 'maxilla_landmark.txt'), delimiter=',')
            if np.isnan(lmk_pre_mand).any():
                lmk_pre_mand = np.genfromtxt(pjoin(all_cases_root, case_name, 'mandible_landmark.txt'), delimiter=',')
            if np.isnan(lmk_pre_skin).any():
                lmk_pre_skin = np.genfromtxt(pjoin(all_cases_root, case_name, 'skin_landmark.txt'), delimiter=',')

            ff_horizontal_normal = np.cross(lmk_pre_maxi[15] - lmk_pre_maxi[9], lmk_pre_maxi[16] - lmk_pre_maxi[10])

            img_pre = load_nifti(glob_root(pjoin(all_cases_root, case_name, '*-pre.nii.gz'))[0])
            seg_pre = threshold_image(img_pre, all_threshold)
            skin_pre = mask_to_object(seg_pre)
            seg_pre_bone = threshold_image(img_pre, bone_threshold)
            bone_pre = mask_to_object(seg_pre_bone)


            lmk_post = load_landmark(pjoin(lmk_root, case_name, 'skin-post-23.csv'))
            img_post = load_nifti(glob_root(pjoin(all_cases_root, case_name, '*-post.nii.gz'))[0])
            seg_post = threshold_image(img_post, all_threshold)
            skin_post = mask_to_object(seg_post)

            t = np.genfromtxt(pjoin(all_cases_root, case_name, case_name+'.tfm'))
            skin_post = transform_polydata(skin_post, t)
            lmk = np.array(list(lmk_post.values()))
            lmk = (np.hstack((lmk, np.ones((lmk.shape[0],1)))) @ t.T)[:,:3].tolist()
            lmk_post = dict(zip(lmk_post.keys(), lmk))

            origin = lmk_pre_mand[12]/2 + lmk_pre_mand[13]/2 # midpoint of two bony gonion superior
            normal = np.array(lmk_pre["Pog'"])-origin # parallel to chin line
            skin_pre_cut_cut = cut_polydata(skin_pre, np.array(lmk_pre['Prn']) + [0,0,10], ff_horizontal_normal)
            skin_pre_cut_cut = cut_polydata(skin_pre_cut_cut, origin, normal)
            skin_pre_cut_cut = cut_polydata(skin_pre_cut_cut, lmk_pre["C"], np.cross(lmk_pre_mand[13]-lmk_pre_mand[22],lmk_pre_mand[12]-lmk_pre_mand[22]))
            skin_pre_cut = cut_polydata(skin_pre, origin, [0,-1,0])
            skin_pre_cut = cut_polydata(skin_pre_cut, lmk_pre["Gb'"], [0,0,-1])
            skin_pre_cut = cut_polydata(skin_pre_cut, lmk_pre["C"], np.cross(lmk_pre_mand[13]-lmk_pre_mand[22],lmk_pre_mand[12]-lmk_pre_mand[22]))
            skin_post_cut_cut = cut_polydata(skin_post, np.array(lmk_pre['Prn']) + [0,0,10], ff_horizontal_normal)
            skin_post_cut_cut = cut_polydata(skin_post_cut_cut, origin, normal)
            skin_post_cut_cut = cut_polydata(skin_post_cut_cut, lmk_pre["C"], np.cross(lmk_pre_mand[13]-lmk_pre_mand[22],lmk_pre_mand[12]-lmk_pre_mand[22]))
            skin_post_cut = cut_polydata(skin_post, origin, [0,-1,0])
            skin_post_cut = cut_polydata(skin_post_cut, lmk_pre["Gb'"], [0,0,-1])
            skin_post_cut = cut_polydata(skin_post_cut, lmk_pre["C"], np.cross(lmk_pre_mand[13]-lmk_pre_mand[22],lmk_pre_mand[12]-lmk_pre_mand[22]))


            write_polydata(skin_pre, pjoin(case_dir, 'skin-pre.stl')) 
            write_polydata(skin_post, pjoin(case_dir, 'skin-post.stl'))
            write_polydata(skin_pre_cut_cut, pjoin(case_dir, 'pre_skin_mesh_ct.stl'))
            write_polydata(skin_pre_cut, pjoin(case_dir, 'pre_soft_tissue_ct.stl'))
            write_polydata(skin_post_cut_cut, pjoin(case_dir, 'post_soft_tissue_ct.stl'))
            write_polydata(skin_post_cut, pjoin(case_dir, 'post_skin_mesh_ct.stl'))
            write_landmark(lmk_pre, pjoin(case_dir, 'skin-pre-23.csv'))
            write_landmark(lmk_post, pjoin(case_dir, 'skin-post-23.csv'))
 

        #__________________________________________________________________________________________#
        # copy inp mesh and segments, generate hex_skin, hex_bone for cutting, calculate pre->post registration for each segment
        

        segs = ('di','diL','diR','le')
        if os.path.exists(pjoin(mesh_root, case_name, 'pre_gen.stl')):
            segs += 'gen', 
            
        if override or not all(isfile(pjoin(case_dir, x)) for x in ['hexmesh_open.inp'] + ['pre_'+s+'.stl' for s in segs] ):

            shutil.copy(pjoin(mesh_root, case_name, 'hexmesh_open.inp'), case_dir)

            stl_dir = pjoin(mesh_root, case_name, 'stl')
            if not isdir(stl_dir):
                stl_dir = input('specify stl dir: ')
            if isdir(stl_dir):
                shutil.move(stl_dir, case_dir)

            for s in segs:
                shutil.copy(pjoin(mesh_root, case_name, 'pre_'+s+'.stl'), case_dir)
                if os.path.exists(pjoin(mesh_root, case_name, 'pre_'+s+'.tfm')):
                    shutil.copy(pjoin(mesh_root, case_name, 'pre_'+s+'.tfm'), case_dir)


        if override or not all(isfile(pjoin(case_dir, 'movement_'+s+'.tfm')) for s in segs):
            stl_dir = pjoin(case_dir, 'stl')
            if isdir(stl_dir):
                for s in segs:
                    stl_name_pre = [f for f in os.listdir(stl_dir) if re.search('.*pre_'+s+'[_]*[0-9]*'+'.stl', f)]
                    stl_name_post = [f for f in os.listdir(stl_dir) if re.search('.*post_'+s+'[_]*[0-9]*'+'.stl', f)]
                    print('found', *stl_name_pre, '/', *stl_name_post)
                    while len(stl_name_pre) != 1:
                        stl_name_pre = glob_root(input(f'type in pre stl {s} segment name in {stl_dir} : '), root_dir=stl_dir)
                    while len(stl_name_post) != 1:
                        stl_name_post = glob_root(input(f'type in post stl {s} segment name for {stl_dir} : '), root_dir=stl_dir)
                    register(
                        pjoin(stl_dir,stl_name_pre[0]), 
                        pjoin(stl_dir, stl_name_post[0]), 
                        pjoin(case_dir, 'movement_'+s+'.tfm')
                        )      
            else:
                print('found no stl segment')


        if override or not os.path.exists(pjoin(case_dir, 'hex_skin.stl')) or not os.path.exists(pjoin(case_dir, 'hex_bone.stl')):
            nodes, elems = read_inp(pjoin(case_dir, 'hexmesh_open.inp'))
            # seed = seed_grid(nodes, elems)
            node_grid = calculate_node_grid(nodes, elems)
            g3d = grid_3d(node_grid)
            g = g3d[:,0,:]
            faces = np.vstack((
                g[:-1,:-1].flat,
                g[1:,:-1].flat,
                g[1:,1:].flat,
                g[:-1,1:].flat,            
            )).T

            nodes = np.ascontiguousarray(nodes)
            faces = np.ascontiguousarray(faces).astype(np.int64)
            polyd = vtkPolyData()
            polyd.SetPoints(numpy_to_vtkpoints_(nodes))
            polyd.SetPolys(numpy_to_vtkpolys_(faces))
            tri = vtkTriangleFilter()
            tri.SetInputData(polyd)
            tri.Update()
            polyd = tri.GetOutput()
            write_polydata(flip_normal_polydata(polyd), pjoin(case_dir, 'hex_bone.stl'))

            g = g3d[:,-1,:]
            faces = np.vstack((
                g[:-1,:-1].flat,
                g[1:,:-1].flat,
                g[1:,1:].flat,
                g[:-1,1:].flat,            
            )).T
            faces = np.ascontiguousarray(faces).astype(np.int64)
            polyd = vtkPolyData()
            polyd.SetPoints(numpy_to_vtkpoints_(nodes))
            polyd.SetPolys(numpy_to_vtkpolys_(faces))
            tri = vtkTriangleFilter()
            tri.SetInputData(polyd)
            tri.Update()
            polyd = tri.GetOutput()
            write_polydata(polyd, pjoin(case_dir, 'hex_skin.stl'))


        if inspection:

            skin_pre_cut_cut = read_polydata(pjoin(case_dir, 'pre_skin_mesh_ct.stl'))
            skin_pre_cut = read_polydata(pjoin(case_dir, 'pre_soft_tissue_ct.stl'))
            skin_post_cut = read_polydata(pjoin(case_dir, 'post_skin_mesh_ct.stl'))
            skin_post_cut_cut = read_polydata(pjoin(case_dir, 'post_soft_tissue_ct.stl'))
            lmk_pre = load_landmark(pjoin(case_dir, 'skin-pre-23.csv'))
            lmk_post = load_landmark(pjoin(case_dir, 'skin-post-23.csv'))
            stls = {}
            t = {}
            post_stls = {}
            for s in segs:
                stls[s] = read_polydata(pjoin(case_dir, 'pre_'+s+'.stl'))
                t[s] = np.genfromtxt(pjoin(case_dir, 'movement_'+s+'.tfm'))
                stls[s] = transform_polydata(stls[s],t[s])
                stl_dir = pjoin(case_dir,'stl')
                stl_name_post = [f for f in os.listdir(stl_dir) if re.search('.*post_'+s+'[_]*[0-9]*'+'.stl', f)]
                while len(stl_name_post) != 1:
                    stl_name_post = glob_root(input(f'type in post stl {s} segment name for {stl_dir} : '), root_dir=stl_dir)
                post_stls[s] = read_polydata(pjoin(stl_dir, stl_name_post[0]))
            show_actors( case_name, [
                        polydata_actor(skin_pre_cut_cut, Color='LightYellow', Opacity=.5),
                        polydata_actor(skin_post_cut_cut, Color='Pink', Opacity=.5),
                        polydata_actor(landmark_polydata(lmk_pre), Color='Silver'),
                        polydata_actor(landmark_polydata(lmk_post), Color='Cyan')
                       ] 
                       + [polydata_actor( s, Color=colors.GetColor3d('Silver')) for s in stls.values()] 
                       + [polydata_actor( s, Color=colors.GetColor3d('Cyan')) for s in post_stls.values()]
            ) 


        if not isfile(pjoin(case_dir,'upper_lip_node_index.txt')) or not isfile(pjoin(case_dir,'lower_lip_node_index.txt')):
            pass

            # nodes, elems = read_inp(pjoin(case_dir, 'hexmesh_open.inp'))
            # node_grid = calculate_node_grid(nodes, elems)
            # _, cols = np.mgrid[range(elems.shape[0]), range(elems.shape[1])]
            # lip_ind = lip_node_index(node_grid)
            # id = np.isin(elems, lip_ind)
            # lip_ind, side = elems[id], seed()[cols[id], 2]
            # lip_upper, lip_lower = np.unique(lip_ind[side>0]), np.unique(lip_ind[side<0])
            # lip_upper = lip_upper[np.argsort(node_grid[lip_upper,0])]
            # lip_upper = lip_upper[np.argsort(node_grid[lip_upper,1])].reshape(6,-1).T
            # lip_lower = lip_lower[np.argsort(node_grid[lip_lower,0])]
            # lip_lower = lip_lower[np.argsort(node_grid[lip_lower,1])].reshape(6,-1).T




    '''
    lower_lip_node_index =

        47157       47124       47091       47058       46992       46993
        47158       47125       47092       47059       46994       46995
        47159       47126       47093       47060       46996       46997
        47160       47127       47094       47061       46998       46999
        47161       47128       47095       47062       47000       47001
        47162       47129       47096       47063       47002       47003
        47163       47130       47097       47064       47004       47005
        47164       47131       47098       47065       47006       47007
        47165       47132       47099       47066       47008       47009
        47166       47133       47100       47067       47010       47011
        47167       47134       47101       47068       47012       47013
        47168       47135       47102       47069       47014       47015
        47169       47136       47103       47070       47016       47017
        47170       47137       47104       47071       47018       47019
        47171       47138       47105       47072       47020       47021
        47172       47139       47106       47073       47022       47023
        47173       47140       47107       47074       47024       47025
        47174       47141       47108       47075       47026       47027
        47175       47142       47109       47076       47028       47029
        47176       47143       47110       47077       47030       47031
        47177       47144       47111       47078       47032       47033
        47178       47145       47112       47079       47034       47035
        47179       47146       47113       47080       47036       47037
        47180       47147       47114       47081       47038       47039
        47181       47148       47115       47082       47040       47041
        47182       47149       47116       47083       47042       47043
        47183       47150       47117       47084       47044       47045
        47184       47151       47118       47085       47046       47047
        47185       47152       47119       47086       47048       47049
        47186       47153       47120       47087       47050       47051
        47187       47154       47121       47088       47052       47053
        47188       47155       47122       47089       47054       47055
        47189       47156       47123       47090       47056       47057
        
    upper_lip_node_index =

        31978       31889       31800       31711       31560       31561
        31979       31890       31801       31712       31562       31563
        31980       31891       31802       31713       31564       31565
        31981       31892       31803       31714       31566       31567
        31982       31893       31804       31715       31568       31569
        31983       31894       31805       31716       31570       31571
        31984       31895       31806       31717       31572       31573
        31985       31896       31807       31718       31574       31575
        31986       31897       31808       31719       31576       31577
        31987       31898       31809       31720       31578       31579
        31988       31899       31810       31721       31580       31581
        31989       31900       31811       31722       31582       31583
        31990       31901       31812       31723       31584       31585
        31991       31902       31813       31724       31586       31587
        31992       31903       31814       31725       31588       31589
        31993       31904       31815       31726       31590       31591
        31994       31905       31816       31727       31592       31593
        31995       31906       31817       31728       31594       31595
        31996       31907       31818       31729       31596       31597
        31997       31908       31819       31730       31598       31599
        31998       31909       31820       31731       31600       31601
        31999       31910       31821       31732       31602       31603
        32000       31911       31822       31733       31604       31605
        32001       31912       31823       31734       31606       31607
        32002       31913       31824       31735       31608       31609
        32003       31914       31825       31736       31610       31611
        32004       31915       31826       31737       31612       31613
        32005       31916       31827       31738       31614       31615
        32006       31917       31828       31739       31616       31617
        32007       31918       31829       31740       31618       31619
        32008       31919       31830       31741       31620       31621
        32009       31920       31831       31742       31622       31623
        32010       31921       31832       31743       31624       31625        
    '''




