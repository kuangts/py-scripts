#!/usr/bin/env python

import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkPlaneSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch, vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkCommonCore import vtkPoints, vtkMath
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPlane
from vtkmodules.vtkCommonTransforms import vtkTransform, vtkMatrixToLinearTransform
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkClipPolyData, vtkPolyDataNormals 
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkCellPicker,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkInteractionWidgets import vtkImplicitPlaneRepresentation,vtkImplicitPlaneWidget2
import vtk
from vtk import vtkPolyData
from basic import Poly, Transform
from register import procrustes
from scipy.spatial import KDTree
from time import perf_counter as tic
from numpy import all, any, eye, sum, mean, sort, unique, bincount, isin, exp, inf
from scipy.spatial import KDTree, distance_matrix
from numpy.linalg import svd, det, solve

colors = vtkNamedColors()

class Poly(vtkPolyData):
    
    def update_normals(self):
        if hasattr(self, 'normals'):
            self.normals.Update()
        else:
            normals = vtkPolyDataNormals()
            normals.SetComputePointNormals(True)
            normals.SetComputeCellNormals(True)
            normals.SetConsistency(False)
            normals.SetInputData(self)
            normals.Update()
            self.normals = normals

    @property
    def F(self):
        return np.reshape(vtk_to_numpy(self.GetPolys().GetData()), (-1,4))[:,1:]

    @property
    def V(self):
        return vtk_to_numpy(self.GetPoints().GetData())

    @V.setter
    def V(self, val):
        pts = self.GetPoints()
        pts.SetData(numpy_to_vtk(val))
        self.update_normals()

    @property
    def VN(self):
        self.update_normals()        
        return vtk_to_numpy(self.normals.GetOutput().GetPointData().GetNormals())

    @property
    def FN(self):
        self.update_normals()        
        return vtk_to_numpy(self.normals.GetOutput().GetCellData().GetNormals())

    @property
    def mean_edge_len(self):
        return np.mean(((self.V[self.F[:,[0,1,2]],:]-self.V[self.F[:,[1,2,0]],:])**2).sum(axis=-1)**.5)

def nicp(SRC, TAR, iterations=20):
    
    pace = 1/5 # cut distance by this portion each iteration, in concept
    # from icp import icp # enable icp
    v_src = SRC.V
    vn_src = SRC.VN

    # nearest-neighbor trees for query
    tar_nn = lambda x,k=1,tar_tree=KDTree(TAR.V): tar_tree.query(x,k=k)[1].flatten()
    src_nn = lambda x,k=1:KDTree(SRC.V).query(x,k=k)[1].flatten() # late binding, updating src_tree has effect
    target_normal_nn = lambda x,k=1,target_normal_tree=KDTree(np.hstack((TAR.V,TAR.VN*SRC.mean_edge_len))): \
         target_normal_tree.query(x,k=k)
    source_normal_nn = lambda x,k=1: KDTree(np.hstack((SRC.V,SRC.VN*SRC.mean_edge_len))).query(x,k=k)
    
    mm, MM = SRC.V.min(axis=0), SRC.V.max(axis=0)
    mindim = min(MM - mm)
    kernel = np.linspace(1,1.5,iterations)[::-1]
    nrseeding = (10**np.linspace(2.1,2.4,iterations)).round()
    kk = 12+iterations
    T = Transform(SRC.V.shape[0]) # reused   

    for i,ker in enumerate(kernel):
        side = mindim/nrseeding[i]**(1/3)
        seedingmatrix = np.concatenate(
            np.meshgrid(
                np.arange(mm[0], MM[0], side),
                np.arange(mm[1], MM[1], side), 
                np.arange(mm[2], MM[2], side), 
                indexing='ij')
            ).reshape(3,-1).T
        # seedingmatrix = SRC.V.copy() # time is ~linear to seedingmatrix size

        D = distance_matrix(SRC.V, seedingmatrix)
        IDX1 = tar_nn(SRC.V)
        IDX2 = src_nn(TAR.V)
        x1 = ~isin(IDX1, TAR.E)
        x2 = ~isin(IDX2, SRC.E)
        sourcepartial = SRC.V[x1,:]
        targetpartial = TAR.V[x2,:]
        _, IDXS = KDTree(targetpartial).query(sourcepartial)
        _, IDXT = KDTree(sourcepartial).query(targetpartial)
        vectors = np.vstack((targetpartial[IDXS]-sourcepartial, targetpartial-sourcepartial[IDXT]))

        ################### Gaussian RBF ################
        D_par = np.vstack((D[x1,:], D[x1,:][IDXT,:]))
        basis = exp(-1/(2*D_par.mean()**ker) * D_par**2) 
        modes = solve( basis.T @ basis + 0.001*eye(basis.shape[1]), basis.T @ vectors )
        SRC.V += exp(-1/(2*D.mean()**ker)*D**2) @ modes * pace


        ################### locally rigid deformation ################
        k = kk-i-1
        arr = np.hstack((SRC.V, SRC.VN*SRC.mean_edge_len))
        Dsource, IDXsource = source_normal_nn(arr, k=k)
        Dtarget, IDXtarget = target_normal_nn(arr, k=3)

        targetV = TAR.V.copy()
        if len(TAR.E): # very important for mesh size difference
            for nei in range(3):
                correctionfortargetholes = isin(IDXtarget[:,nei], TAR.E)
                IDXtarget[correctionfortargetholes,nei] = targetV.shape[0] + np.arange(sum(correctionfortargetholes))
                Dtarget[correctionfortargetholes,nei] = 1e-5
                targetV = np.vstack((targetV, SRC.V[correctionfortargetholes,:]))

        Wtarget = (1 - Dtarget/(Dtarget.sum(axis=1,keepdims=True)))/(3-1)
        targetV = sum(Wtarget[...,None]*targetV[IDXtarget,:], axis=1)
        *_, T[...] = procrustes(SRC.V[IDXsource,:], targetV[IDXsource,:], scaling=False)
        Wsource = (1 - Dsource/(Dsource.sum(axis=1,keepdims=True)))/(k-1)
        T[...] = sum(Wsource[...,None,None] * T[IDXsource,:,:], axis=1)

        SRC.V = v_src + sum((SRC.V[:,None,:].transform(T).squeeze() - v_src) * vn_src, axis=1, keepdims=True) * vn_src * pace

        # end of iteration, keep result for next
        v_src[...] = SRC.V
        vn_src[...] = SRC.VN

    return SRC








def polydata_to_numpy(polydata):
    numpy_nodes = vtk_to_numpy(polydata.GetPoints().GetData())
    numpy_faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1,4)[:,1:].astype(int)
    return {
        'nodes':numpy_nodes,
        'faces':numpy_faces
    }

def numpy_to_polydata(nodes, faces):
    polydata = vtkPolyData()
    points = vtkPoints()
    points.SetData(numpy_to_vtk(nodes))
    polydata.SetPoints(points)
    polys = vtkCellArray()
    polys.SetData(3,numpy_to_vtk(faces.ravel()))
    polydata.SetPolys(polys)
    return polydata

def vtkMatrix4x4_to_numpy(mat4x4):
    t = np.eye(4)
    mat4x4.DeepCopy(t.ravel(), mat4x4)
    return t

def numpy_to_vtkMatrix4x4(np4x4):
    mat = vtkMatrix4x4()
    mat.DeepCopy(np4x4.ravel())        
    return mat




def test_register(skin_path=r'C:\data\midsagittal\skin_smooth_10mm.stl'):

    def key_press_event(obj, event):
        key = obj.GetInteractor().GetKeySym()
        if key == 'r':
            find_plane()
        if key == 'x':
            set_plane(
                origin = (201.6504424228966, 171.96995704132829, 130.60506472392518),
                normal = (0.9998767781054699, 0.015686128773640704, -0.0006115304747014283)
            )
        if key.isdigit():
            if key in _planes:
                set_plane(_planes[key]['origin'],_planes[key]['normal'])
            else:
                record_plane(key)

    def set_plane(origin, normal):
        plane_rep.SetOrigin(*origin)
        plane_rep.SetNormal(*normal)
        plane_rep.GetPlane(plane)
        T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -vtkMath.Dot(origin, normal) ]])
        _T.DeepCopy(T.ravel())
        half0.Update()
        half1.Update()
        half1m.Update()
        renwin.Render()

    def find_plane():
        h0 = half0.GetOutput()
        h1 = half1.GetOutput()
        h1m = vtkPolyData()
        h1m.DeepCopy(h1)
        tfm = vtkTransform()
        tfm.SetMatrix(_T)
        tfm.Update()
        pm = vtkPoints()
        tfm.TransformPoints(h1m.GetPoints(), pm)
        h1m.GetPoints().DeepCopy(pm)

        # tar_np = polydata_to_numpy(h0)
        # tar = Poly(V=tar_np['nodes'],F=tar_np['faces'])
        # src_np = polydata_to_numpy(h1)

        src = Poly()
        src.DeepCopy(h1)
        tar = Poly()
        tar.DeepCopy(h0)
        reg = nicp(src, tar, iterations=20)
        points = src_np['nodes']/2 + reg_np['nodes']/2
        print(points)
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points))
        origin, normal = [0.]*3, [0.]*3
        vtkPlane.ComputeBestFittingPlane(vtk_points, origin, normal)
        set_plane(origin, normal)

        nicp_result.GetPoints().SetData(numpy_to_vtk(reg_np['nodes']))



    ren = vtkRenderer()
    ren.SetBackground(colors.GetColor3d('PaleGoldenrod'))
    renwin = vtkRenderWindow()
    renwin.AddRenderer(ren)
    renwin.SetWindowName('InteractorStyleTrackballCamera')

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)
    style = vtkInteractorStyleTrackballCamera()
    style.AddObserver('KeyPressEvent', key_press_event)
    iren.SetInteractorStyle(style)

    _T = vtkMatrix4x4()
    plane = vtkPlane()    
    _planes = {}

    reader = vtkSTLReader()
    reader.SetFileName(skin_path)
    reader.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(reader.GetOutputPort())
    cleaner.Update()

    #### INITIAL PLANE AND CAMERA ####
    xmin, xmax, ymin, ymax, zmin, zmax = cleaner.GetOutput().GetBounds()
    xdim, ydim, zdim = xmax-xmin, ymax-ymin, zmax-zmin
    xmid, ymid, zmid = xmin/2+xmax/2, ymin/2+ymax/2, zmin/2+zmax/2
    plane_rep = vtkImplicitPlaneRepresentation()
    plane_rep.SetWidgetBounds(xmin-xdim*.2,xmax+xdim*.2,ymin-ydim*.2,ymax+ydim*.2,zmin-zdim*.2,zmax+zdim*.2)
    plane_rep.SetDrawOutline(False)
    plane_rep.SetOrigin(xmid, ymin-ydim*.1, zmid)
    plane_rep.SetNormal(1., 0., 0.)
    plane_rep.GetPlane(plane)
    plane_widget = vtkImplicitPlaneWidget2()
    plane_widget.SetRepresentation(plane_rep)
    plane_widget.SetInteractor(iren)
    plane_widget.On()

    cam = ren.GetActiveCamera()
    cam.SetPosition(xmid, ymid-ydim*4, zmid)
    cam.SetFocalPoint(xmid, ymid, zmid)
    cam.SetViewUp(0., 0., 1.)

    #### REFLECTION BY PLANE ####
    origin, normal = plane_rep.GetOrigin(), plane_rep.GetNormal()
    T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -vtkMath.Dot(origin, normal) ]])
    _T.DeepCopy(T.ravel())
    mirror = vtkMatrixToLinearTransform()
    mirror.SetInput(_T)
    mirror.Update()

    clipper = vtkClipPolyData()
    clipper.SetClipFunction(plane)
    clipper.SetInputConnection(cleaner.GetOutputPort())
    clipper.GenerateClippedOutputOn()
    clipper.Update()

    half0 = vtk.vtkCleanPolyData()
    half0.SetInputConnection(clipper.GetOutputPort(0))
    half0.Update()

    half1 = vtk.vtkCleanPolyData()
    half1.SetInputConnection(clipper.GetOutputPort(1))
    half1.Update()

    half1m = vtkTransformPolyDataFilter()
    half1m.SetTransform(mirror)
    half1m.SetInputConnection(half1.GetOutputPort())
    half1m.Update()

    find_plane()

    #### MIRRORED HALF MODEL ####
    mapper_half_mirrored = vtkPolyDataMapper()
    mapper_half_mirrored.SetInputConnection(half1m.GetOutputPort())
    actor_half_mirrored = vtkActor()
    actor_half_mirrored.SetMapper(mapper_half_mirrored)
    actor_half_mirrored.GetProperty().SetColor(colors.GetColor3d('Yellow'))
    ren.AddActor(actor_half_mirrored)

    #### NICP MODEL ####
    nicp_result = vtkPolyData()
    nicp_result.DeepCopy(half1m.GetOutput())
    nicp_mapper = vtkPolyDataMapper()
    nicp_mapper.SetInputDataObject(nicp_result)
    nicp_mapper.Update()
    actor_incp = vtkActor()
    actor_incp.SetMapper(nicp_mapper)
    actor_incp.GetProperty().SetColor(colors.GetColor3d('Green'))
    actor_incp.GetProperty().SetOpacity(0)
    ren.AddActor(actor_incp)

    #### HALF0 ####
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(half0.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarRange(0.,10.)
    actor_left = vtkActor()
    actor_left.SetMapper(mapper)
    actor_left.GetProperty().SetColor(colors.GetColor3d('Red'))
    ren.AddActor(actor_left)

    #### HALF1 ####
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(half1.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarRange(0.,10.)
    actor_right = vtkActor()
    actor_right.SetMapper(mapper)
    actor_right.GetProperty().SetColor(colors.GetColor3d('Blue'))
    ren.AddActor(actor_right)

    iren.Initialize()
    renwin.Render()
    iren.Start()


if __name__ == '__main__':
    test_register()
