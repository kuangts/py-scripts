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
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData
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

colornames = [
            "IndianRed",
            "Lavender",
            "Aqua",
            "Cornsilk",
            "LightSalmon",
            "Gold",
            "GreenYellow",
            "Pink",
            "Gainsboro",
        ]

colors = vtkNamedColors()


def load_nifti(file):
    src = vtkNIFTIImageReader()
    src.SetFileName(file)
    src.Update()
    matq = vtkmatrix4x4_to_numpy(src.GetQFormMatrix())
    mats = vtkmatrix4x4_to_numpy(src.GetSFormMatrix())
    assert np.allclose(matq, mats), 'nifti image qform and sform matrices not the same'
    origin = matq[:3,:3] @ matq[:3,3]
    polyd = src.GetOutput()
    polyd.SetOrigin(origin.tolist())
    return polyd


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
    
    geometry.Update()

    return geometry.GetOutput()


def write_polydata(polyd, file):
    writer = vtkSTLWriter()
    writer.SetInputData(polyd)
    writer.SetFileName(file)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def read_polydata(file):
    reader = vtkSTLReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()


def translate_polydata(polyd, t):
    T = vtkTransform()
    T.Translate(t)
    Transform = vtkTransformPolyDataFilter()
    Transform.SetTransform(T)
    Transform.SetInputData(polyd)
    Transform.Update()
    return Transform.GetOutput()


def transform_polydata(polyd, t):
    T = vtkTransform()
    M = vtkMatrix4x4()
    arr = vtkmatrix4x4_to_numpy(M)
    arr[...] = t
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


def show_polydata(case_name, polyds, properties=None):
    
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

    for i,m in enumerate(polyds):

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(m)
        actor = vtkActor()
        actor.SetMapper(mapper)
        if properties is not None and properties[i] is not None:
            propty = actor.GetProperty()
            for pk,pv in properties[i].items():
                getattr(propty,'Set'+pk).__call__(pv)
        renderer.AddActor(actor)

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
    input.SetPoints(numpy_to_vtkpoints(lmk_np_nx3))

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


def write_landmark(lmk, file):
    with open(file, 'w', newline='') as f:
        csv.writer(f).writerows([[k,*v] for k,v in lmk.items()])


def check_case(case_dir):

    lmk_pre = load_landmark(pjoin(case_dir, 'skin-pre-23.csv'))
    skin_pre = read_polydata(pjoin(case_dir, 'skin-pre.stl'))

    lmk_post = load_landmark(pjoin(case_dir, 'skin-post-23.csv'))
    skin_post = read_polydata(pjoin(case_dir, 'skin-post.stl'))

    show_polydata(os.path.basename(os.path.normpath(case_name)), 
              [skin_pre, skin_post, landmark_polydata(lmk_pre), landmark_polydata(lmk_post)], 
              [{'Color':colors.GetColor3d(colornames[i])} for i in range(4)])



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
    t = vtkmatrix4x4_to_numpy(T.GetMatrix())
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


def calculate_node_grid(nodes, elems, seed=None): # alters self, to be subclassed
    ng = np.empty(nodes.shape)
    ng[:] = np.nan

    seed = np.array([
        [+1,+1,+1],
        [+1,-1,+1],
        [-1,-1,+1],
        [-1,+1,+1],
        [+1,+1,-1],
        [+1,-1,-1],
        [-1,-1,-1],
        [-1,+1,-1],
    ])
    # seed = seed[[0, 3, 1, 2, 4, 7, 5, 6],:]
    # seed = seed*[1,-1,1]
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



if __name__=='__main__':

    box = r'C:\Users\xiapc\Box'
    completed_cases_root = rf'{box}\RPI\data\FEM_DL'
    recent_cases_root = rf'{box}\Facial Prediction_DK_TK\recent_cases'
    all_cases_root = rf'{box}\RPI\data\pre-post-paired-unc40-send-1122'
    skin_lmk_root = rf'{box}\RPI\data\pre-post-paired-soft-tissue-lmk-23'
    working_root = rf'P:\20230428'
    lmk_root = rf'{box}\RPI\data\pre-post-paired-soft-tissue-lmk-23'
    segment_root = rf'C:\Users\tians\OneDrive\meshes'

    do_checking = False

    list_of_files = ['hexmesh_open.inp', 'pre_skin_mesh_ct.stl', 'pre_soft_tissue_ct.stl', 'post_skin_mesh_ct.stl', 'movement_di.tfm', 'movement_diL.tfm', 'movement_diR.tfm', 'movement_le.tfm', 'pre_di.stl','pre_diL.stl','pre_diR.stl','pre_le.stl']

    cases_40 = glob.glob('n*', root_dir=all_cases_root)
    cases_done = glob.glob('n*_ActualSurgery*', root_dir=completed_cases_root)
    cases_done = [f'n{int(x[1:3]):04}' for x in cases_done]
    cases_recent = glob.glob('n*', root_dir=recent_cases_root)
    cases_recent_not_completed = [x for x in cases_recent if x not in cases_done]
    print(f'{len(cases_recent_not_completed)} recent cases not completed:', *cases_recent_not_completed)
    cases_done.sort()
    cases_working = [x for x in cases_40 if x not in cases_done and x not in cases_recent_not_completed]


    try:
        cases_working.remove('n0042')
    except:
        pass
    print(f'working on {len(cases_working)} cases:', *cases_working)

    os.makedirs(working_root, exist_ok=True)
    for case_name in cases_working:

        case_dir = pjoin(working_root, case_name)
        if all([isfile(pjoin(case_dir, x)) for x in list_of_files]):
            print(f'{case_name} is completed.')
            continue
        os.makedirs(case_dir, exist_ok=True)

        #__________________________________________________________________________________________#
        # first processing, write pre and registered post stl and lmk, cut stl

        if not all([isfile(pjoin(case_dir, x)) for x in ('skin-pre.stl', 'skin-post.stl', 'skin-pre-23.csv', 'skin-post-23.csv', 'pre_skin_mesh_ct.stl', 'pre_soft_tissue_ct.stl', 'post_skin_mesh_ct.stl')]):

            lmk_pre = load_landmark(pjoin(lmk_root, case_name, 'skin-pre-23.csv'))
            img_pre = load_nifti(glob.glob(pjoin(all_cases_root, case_name, '*-pre.nii.gz'))[0])
            seg_pre = threshold_image(img_pre, (-1000, 300))
            skin_pre = mask_to_object(seg_pre)

            prn = np.array(lmk_pre['Prn'])
            zyr = np.array(lmk_pre["Zy'-R"])
            zyl = np.array(lmk_pre["Zy'-L"])

            origin = (prn+zyr+zyl)/3
            normal = np.cross(zyr-prn, zyl-prn)
            skin_pre_cut = cut_polydata(skin_pre, origin, normal)

            go = np.array(lmk_pre["Go'-R"])/2 + np.array(lmk_pre["Go'-L"])/2
            skin_pre_cut = cut_polydata(skin_pre_cut, go, [normal[0], normal[2],-normal[1]])

            lmk_post = load_landmark(pjoin(lmk_root, case_name, 'skin-post-23.csv'))
            img_post = load_nifti(glob.glob(pjoin(all_cases_root, case_name, '*-post.nii.gz'))[0])
            seg_post = threshold_image(img_post, (-1000, 300))
            skin_post = mask_to_object(seg_post)

            t = np.genfromtxt(pjoin(all_cases_root, case_name, case_name+'.tfm'))
            skin_post = transform_polydata(skin_post, t)
            lmk = np.array(list(lmk_post.values()))
            lmk = (np.hstack((lmk, np.ones((lmk.shape[0],1)))) @ t.T)[:,:3].tolist()
            lmk_post = dict(zip(lmk_post.keys(), lmk))

            prn = np.array(lmk_post['Prn'])
            zyr = np.array(lmk_post["Zy'-R"])
            zyl = np.array(lmk_post["Zy'-L"])

            origin = (prn+zyr+zyl)/3
            normal = np.cross(zyr-prn, zyl-prn)
            skin_post_cut = cut_polydata(skin_post, origin, normal)

            go = np.array(lmk_post["Go'-R"])/2 + np.array(lmk_post["Go'-L"])/2
            skin_post_cut = cut_polydata(skin_post_cut, go, [normal[0], normal[2],-normal[1]])

            show_polydata(case_name, [skin_pre_cut, skin_post_cut, landmark_polydata(lmk_pre), landmark_polydata(lmk_post)], [{'Color':colors.GetColor3d(colornames[i])} for i in range(4)])

            write_polydata(flip_normal_polydata(skin_pre), pjoin(case_dir, 'skin-pre.stl')) 
            write_polydata(flip_normal_polydata(skin_post), pjoin(case_dir, 'skin-post.stl'))
            write_polydata(flip_normal_polydata(skin_pre_cut), pjoin(case_dir, 'pre_skin_mesh_ct.stl'))
            write_polydata(flip_normal_polydata(skin_post_cut), pjoin(case_dir, 'post_skin_mesh_ct.stl'))
            write_landmark(lmk_pre, pjoin(case_dir, 'skin-pre-23.csv'))
            write_landmark(lmk_post, pjoin(case_dir, 'skin-post-23.csv'))
 

        #__________________________________________________________________________________________#
        # copy inp mesh and segments, generate hex_skin for cutting, calculate pre->post registration for each segment
        
        segs = ('di','diL','diR','le')
        if not all([isfile(pjoin(case_dir, x)) for x in ['hexmesh_open.inp'] + ['pre_'+s+'.stl' for s in segs] ]):

            shutil.copy(pjoin(segment_root, case_name, 'hexmesh_open.inp'), case_dir)

            for s in segs:
                shutil.copy(pjoin(segment_root, case_name, 'pre_'+s+'.stl'), case_dir)
                if os.path.exists(pjoin(segment_root, case_name, 'pre_'+s+'.tfm')):
                    shutil.copy(pjoin(segment_root, case_name, 'pre_'+s+'.tfm'), case_dir)

            s = 'gen'
            if os.path.exists(pjoin(segment_root, case_name, 'pre_'+s+'.stl')):
                shutil.copy(pjoin(segment_root, case_name, 'pre_'+s+'.stl'), case_dir)
                if os.path.exists(pjoin(segment_root, case_name, 'pre_'+s+'.tfm')):
                    shutil.copy(pjoin(segment_root, case_name, 'pre_'+s+'.tfm'), case_dir)

            s = 'hex_skin.stl'
            if os.path.exists(pjoin(segment_root, case_name, s)):
                shutil.copy(pjoin(segment_root, case_name, s), case_dir)
                print(case_name, 'has hex skin')


        if not os.path.exists(pjoin(case_dir, 'hex_skin.stl')):
            nodes, elems = read_inp(pjoin(case_dir, 'hexmesh_open.inp'))
            # seed = seed_grid(nodes, elems)
            node_grid = calculate_node_grid(nodes, elems)
            print(len(np.unique(node_grid, axis=0)))
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
            polyd.SetPoints(numpy_to_vtkpoints(nodes))
            polyd.SetPolys(numpy_to_vtkpolys(faces))
            tri = vtkTriangleFilter()
            tri.SetInputData(polyd)
            tri.Update()
            polyd = tri.GetOutput()
            write_polydata(polyd, pjoin(case_dir, 'hex_skin.stl'))
            # show_polydata(case_name, [polyd], [{'Color':colors.GetColor3d(colornames[i])} for i in range(1)])
 
        

        if not os.path.exists(pjoin(case_dir, 'pre_skin_mesh_ct.stl')):
            pass
            




        if do_checking:
            check_case(case_dir)




    # extra_cases = glob.glob('n*', root_dir=working_root)
    # completed_cases = glob.glob('n*', root_dir=completed_cases_root)
    # for case_name in extra_cases:
    #     if case_name in completed_cases:
    #         shutil.move(case_dir, pjoin(working_root, 'finished'))

