import glob, os, csv, shutil, re
from os.path import join as pjoin
from os.path import exists as pexists
from os.path import isfile as isfile
from os.path import isdir as isdir
from os.path import basename, dirname, normpath, realpath
from typing import Any
from vtk import vtkRegularPolygonSource, vtkPolygon

import numpy as np
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkParametricFunctionSource
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkTriangleFilter, vtkClipPolyData, vtkPolyDataConnectivityFilter, vtkImplicitPolyDataDistance, vtkAppendPolyData
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkPolyLine, vtkUnstructuredGrid, vtkImplicitSelectionLoop, vtkPointLocator, vtkImplicitDataSet
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkFloatArray
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget, vtkBoxRepresentation, vtkContourWidget
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkTransformFilter, vtkBooleanOperationPolyDataFilter 
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
    vtkRenderer,
    vtkProp3DFollower,
    vtkCoordinate
)
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from vtkmodules.vtkCommonComputationalGeometry import vtkParametricSpline
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkPolyDataNormals
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkSmoothPolyDataFilter, vtkCleanPolyData, vtkFeatureEdges, vtkStripper
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
    vtkPlane,
    vtkBox
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
from vtkmodules.vtkInteractionWidgets import vtkPolygonalSurfacePointPlacer, vtkOrientedGlyphContourRepresentation
from vtkmodules.vtkFiltersModeling import vtkSelectPolyData, vtkRibbonFilter
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData
)
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
from vtkmodules.vtkInteractionWidgets import vtkWidgetEvent

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from vtk_bridge import *

mode = 'auto' # 'manual' or 'auto'
div = 0
# because there is no definition of reference frame, whatsoever
sign = -1 # change between +1 and -1

colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']

colors = vtkNamedColors()

def read_polydata(file):
    reader = vtkSTLReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()


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


def points_polydata(pts):

    if isinstance(pts, dict):
        pts = list(pts.values())

    if not isinstance(pts, vtkPoints):
        pts = share_numpy_to_vtkpoints(np.array(pts))

    input = vtkPolyData()
    input.SetPoints(pts)

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


def strip_edges(polyd):
    edge_filter = vtkFeatureEdges()
    edge_filter.ExtractAllEdgeTypesOff()
    edge_filter.BoundaryEdgesOn()
    edge_filter.SetInputData(polyd)
    stripper = vtkStripper()
    stripper.SetInputConnection(edge_filter.GetOutputPort())
    stripper.Update()
    return stripper.GetOutput()


if __name__=='__main__':

    root = r'.\test'
    polyd_0 = read_polydata(os.path.join(root,r'breast_volume_mm.stl'))
    polyd_1 = read_polydata(os.path.join(root,r'chest_wall_mm.stl'))
    volume = polydata_actor(polyd_0, Color='LightPink')
    chest = polydata_actor(polyd_1, Color='LightCyan')

    vtx = share_vtkpoints_to_numpy(polyd_0.GetPoints())
    fcs = share_vtkpolys_to_numpy(polyd_0.GetPolys())


    if mode == 'auto':
        # volume resembles an oblate ellipsoid
        # reduce the axial/symmetry dimension, triangulate, and strip boundary
        fitter = PCA(n_components=3).fit(vtx)
        vtx_t = fitter.transform(vtx)

        # triangulate 2d projection
        vtx2d = vtx_t.copy()
        vtx2d[:,2] = 0
        polyd2d = vtkPolyData()
        pts = vtk.vtkPoints()
        pts.SetData(share_numpy_to_vtk(vtx2d))
        polyd2d.SetPoints(pts)
        dln = vtk.vtkDelaunay2D()
        dln.SetInputData(polyd2d)
        dln.Update()
        polyd2d = dln.GetOutput()

        # strip edges and 
        polyd2d_edg = strip_edges(polyd2d)
        vtx_edg = share_vtkpoints_to_numpy(polyd2d_edg.GetPoints()).copy()
        edg = share_vtkpolys_to_numpy(polyd2d_edg.GetLines())[0].copy()
        # find indices in original polydata
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vtx2d)
        distances, indices = nbrs.kneighbors(vtx_edg)
        edg = indices[edg].flatten()

        # lin space into 52 (improve later)
        d = vtx[edg[:-1],:] - vtx[edg[1:],:]
        d = np.sum(d**2, axis=1)**.5
        d_cum = np.cumsum([0,*d])
        d_52 = np.linspace(0, d_cum.max(), 53)[:-1]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(d_cum.reshape(-1,1))
        distances, indices = nbrs.kneighbors(d_52.reshape(-1,1))
        vtx_52 = vtx[edg[indices.flatten()],:]
        tan_52_0 = vtx_52-np.vstack((vtx_52[1:], vtx_52[[0],:]))
        tan_52_1 = np.vstack((vtx_52[[-1],:], vtx_52[:-1])) - vtx_52
        tan_52 = tan_52_0/2 + tan_52_1/2
        tan_52 = tan_52/np.sum(tan_52**2, axis=1, keepdims=True)**.5

        # spline might results in shooting rays with no intersection
        # xsp = CubicSpline(d_cum, vtx[edg,0])
        # ysp = CubicSpline(d_cum, vtx[edg,1])
        # zsp = CubicSpline(d_cum, vtx[edg,2])
        # vtx_52 = np.vstack((xsp(d_52), ysp(d_52), zsp(d_52))).T
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(vtx_52[:,0],vtx_52[:,1],vtx_52[:,2])
        # plt.show()

    elif mode=='manual':

        vtx_52 = np.array([[float('nan')]*3]*52)

        def _contour_widget_select(obj, event, vtx=vtx_52):
            polyd = obj.GetRepresentation().GetContourRepresentationAsPolyData()
            if not polyd.GetPoints() or not polyd.GetPoints().GetNumberOfPoints():
                return
            coords = share_vtkpoints_to_numpy(polyd.GetPoints())
            if obj.GetWidgetState() == vtkContourWidget.Manipulate:
                d = np.sum((coords[:-1,:] - coords[1:,:])**2, axis=1)**.5
                d_cum = np.cumsum([0,*d])
                d_52 = np.linspace(0, d_cum.max(), 53)[:-1]
                xsp = CubicSpline(d_cum, coords[:,0])
                ysp = CubicSpline(d_cum, coords[:,1])
                zsp = CubicSpline(d_cum, coords[:,2])
                vtx[...] = np.vstack((xsp(d_52), ysp(d_52), zsp(d_52))).T


        renderer = vtkRenderer()
        renderWindow = vtkRenderWindow()
        renderWindow.SetWindowName('PolyLine')
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(volume)
        renderer.AddActor(chest)
        renderer.SetBackground(colors.GetColor3d('DarkOliveGreen'))
        style = vtkInteractorStyleTrackballCamera()
        style.SetDefaultRenderer(renderer)
        renderWindowInteractor.SetInteractorStyle(style)
        renderWindow.Render()

        # add contour widget to create variabl edg
        contour_widget = vtkContourWidget()
        contour_widget.SetInteractor(renderWindowInteractor)
        contour_widget.AddObserver(vtkWidgetEvent.Select, _contour_widget_select)
        pointPlacer = vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(volume)

        rep = contour_widget.GetRepresentation()
        rep.GetLinesProperty().SetColor(colors.GetColor3d("Crimson"))
        rep.GetLinesProperty().SetLineWidth(3.0)
        rep.SetPointPlacer(pointPlacer)

        contour_widget.EnabledOn()
        renderWindowInteractor.Start()

        # close window to continue

    # lin space into 52 (improve later)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vtx)
    _, edg = nbrs.kneighbors(vtx_52)
    edg = edg.flatten()
    vtx_52 = vtx[edg]
    tan_52_0 = vtx_52-np.vstack((vtx_52[1:], vtx_52[[0],:]))
    tan_52_1 = np.vstack((vtx_52[[-1],:], vtx_52[:-1])) - vtx_52
    tan_52 = tan_52_0/2 + tan_52_1/2
    tan_52 = tan_52/np.sum(tan_52**2, axis=1, keepdims=True)**.5

    # trace from source to these 52 vertices
    fitter = PCA(n_components=3).fit(vtx_52)
    vtx_52_t = fitter.transform(vtx_52)
    bd_box = np.vstack((vtx_52_t.min(axis=0), vtx_52_t.max(axis=0)))
    bd_box = fitter.inverse_transform(bd_box)
    bd_center = bd_box.mean(axis=0)

    obbTree_0 = vtk.vtkOBBTree()
    obbTree_0.SetDataSet(polyd_0)
    obbTree_0.BuildLocator()

    obbTree_1 = vtk.vtkOBBTree()
    obbTree_1.SetDataSet(polyd_1)
    obbTree_1.BuildLocator()

    volume_intercept = np.array([[float('nan')]*3]*52)
    chestwall_intercept = np.array([[float('nan')]*3]*52)
    points = vtkPoints()
    cells = vtkCellArray()
    liga = vtkPolyData()
    liga.SetPoints(points)
    liga.SetLines(cells)

    for i in range(52):
        
        des = vtx_52[i,:]
        rad = bd_center-des
        rad = rad/np.sum(rad**2)**.5
        des = des + .05*(rad) # to make sure there is a hit
        dir = np.cross(tan_52[i], rad)
        src = des + 20*(dir*sign + div*rad)
        des = src + (des-src)*100 # leave enough room

        pts = vtk.vtkPoints()
        cellIds = vtk.vtkIdList()
        if obbTree_0.IntersectWithLine(src, des, pts, cellIds):
            points.InsertNextPoint(pts.GetPoint(pts.GetNumberOfPoints()-1))
        else:
            points.InsertNextPoint([float('nan')]*3)

        pts = vtk.vtkPoints()
        cellIds = vtk.vtkIdList()
        if obbTree_1.IntersectWithLine(src, des, pts, cellIds):
            points.InsertNextPoint(pts.GetPoint(0))
        else:
            points.InsertNextPoint([float('nan')]*3)

        l = vtkPolyLine()
        l.GetPointIds().SetNumberOfIds(2)
        l.GetPointIds().SetId(0,i*2)
        l.GetPointIds().SetId(1,i*2+1)
        cells.InsertNextCell(l)

    liga.Modified()
    # volume = polydata_actor(polyd_0, Color='LightPink')
    # chest = polydata_actor(polyd_1, Color='LightCyan')
    ligaments = polydata_actor(liga, Color='tomato')
    lineends = polydata_actor(points_polydata(points), Color='tomato')

    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('PolyLine')
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.AddActor(volume)
    renderer.AddActor(chest)
    renderer.AddActor(ligaments)
    renderer.AddActor(lineends)
    renderer.SetBackground(colors.GetColor3d('DarkOliveGreen'))
    style = vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindow.Render()
    renderWindowInteractor.Start()

    mat2write = share_vtkpoints_to_numpy(points).reshape(-1,6).tolist()
    with open(os.path.join(root,r'liga_p5_r.csv'), 'w', newline='') as f: 
        writer = csv.writer(f)
        for l in mat2write:
            writer.writerow(l)

