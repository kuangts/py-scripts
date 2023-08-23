#!/usr/bin/env python3
'''
Tianshu Kuang
Houston Methodist Hospital
07/2023
'''

import sys, pkg_resources
try:
    pkg_resources.require(['numpy','vtk'])
except Exception as e:
    sys.exit(e)

import os, csv
from glob import glob
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonDataModel import vtkBox
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPolygon
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.util.numpy_support import numpy_to_vtk
from tools.procrustes import *
# from register import nicp # cannot run without this, don't know why


_colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']


def _numpy_to_vtkpoints(arr:np.ndarray, vtk_pts:vtkPoints=None):
    """wraps numpy_to_vtk() for convenience
    
    for ease of use:
        if vtk_pts is given, perform in-place modification and return None
        else create new vtkPoints instance and return it

    """
    pts = vtkPoints() if vtk_pts is None else vtk_pts
    pts.SetData(numpy_to_vtk(arr))
    return pts if vtk_pts is None else pts.Modified() 


def _points_polydata(pts:vtkPoints, sphere_radius=1.0, return_glyph=False):

    input = vtkPolyData()
    input.SetPoints(pts)

    glyphSource = vtkSphereSource()
    glyphSource.SetRadius(sphere_radius)
    glyphSource.Update()

    glyph3D = vtkGlyph3D()
    glyph3D.GeneratePointIdsOn()
    glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
    glyph3D.SetInputData(input)
    glyph3D.SetScaleModeToDataScalingOff()
    glyph3D.Update()
    if return_glyph:
        return glyph3D.GetOutput(), glyph3D
    return glyph3D.GetOutput()


def _plane_polydata(plane_abcd, bounds):
    intersections = np.empty((18,), dtype=float)
    intersections[...] = float('nan')
    vtkBox.IntersectWithPlane(bounds, -plane_abcd[:-1]*plane_abcd[-1], plane_abcd[:-1], intersections)
    verts = intersections.reshape(-1,3)
    verts = verts[np.any(verts!=0, axis=1) & np.any(~np.isnan(verts), axis=1)]

    polyd = vtkPolyData()
    points = vtkPoints()
    for p in verts:
        points.InsertNextPoint(p)
    cells = vtkCellArray()
    l = vtkPolygon()
    n = points.GetNumberOfPoints()
    l.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        l.GetPointIds().SetId(i,i)
    cells.InsertNextCell(l)
    polyd.SetPoints(points)
    polyd.SetPolys(cells)

    return polyd


def _polydata_actor(polyd:vtkPolyData, mapper=None, **property):
    if mapper is None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polyd)
    actor = vtkActor()
    actor.SetMapper(mapper)
    if property:
        for pk,pv in property.items():
            if pk=='Color':
                if isinstance(pv, int):
                    pv = _colornames[pv]
                if isinstance(pv, str):
                    pv = vtkNamedColors().GetColor3d(pv)
            getattr(actor.GetProperty(),'Set'+pk).__call__(pv)

    return actor


def _render_window(window_title=''):
    renderer = vtkRenderer()
    renderer.SetBackground(.67, .93, .93)

    _render_window = vtkRenderWindow()
    _render_window.AddRenderer(renderer)
    _render_window.SetSize(1000,1500)
    _render_window.SetWindowName(window_title)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(_render_window)

    style = vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)
    return interactor, renderer


def _sigmoid(c1, c2):
    return lambda x: 1-1/(1+np.exp(-c1*(x-c2)))


def test_real_patient():
    weighting_func = _sigmoid(.8,5) 
    subs = glob(r'C:\data\pre-post-paired-soft-tissue-lmk-23\n00*')
    landmarks = []
    for sub in subs[:10]:
        labels = []
        coords = []
        with open(os.path.join(sub, 'skin-pre-23.csv'), 'r') as f:
            for l in csv.reader(f):
                labels.append(l[0])
                coords.append(l[1:])
        coords = np.array(coords, dtype=float)
        labels = np.array(labels, dtype=str)
        ind = np.argsort(labels)
        assert labels.shape[0] == 23, 'missing points'
        labels, coords = labels[ind], coords[ind]
        group = np.arange(labels.shape[0])
        group[['-L' in x and x.replace('-L','-R') in labels for x in labels]] = -1
        group[['-R' in x and x.replace('-R','-L') in labels for x in labels]] = 1
        group[['-L' not in x and '-R' not in x for x in labels]] = 0

        landmarks.append(coords)

    d0, landmarks_registered, _ = iterative_procrustes(*landmarks, weighting_func=None, target_shape_index=0)

    #########################################################################
    # # VISUALIZE THEM ALL
    #########################################################################
    iren, ren = _render_window()        
    for i,lmk in enumerate(landmarks_registered):
        polyd = _points_polydata(_numpy_to_vtkpoints(lmk))
        ren.AddActor(_polydata_actor(polyd, Color=i))
    iren.Start()

    #########################################################################
    # # THE EFFECT OF NAN
    #########################################################################
    for i,lmk in enumerate(landmarks):
        ind = np.random.randint(0,lmk.shape[0],size=(2,))
        lmk[ind,:] = np.nan

    _, landmarks_registered_nan, _ = iterative_procrustes(*landmarks, weighting_func=None, target_shape_index=0)

    for i,(lmk, lmk_nan) in enumerate(zip(landmarks_registered, landmarks_registered_nan)):
        iren, ren = _render_window()
        
        polyd = _points_polydata(_numpy_to_vtkpoints(lmk))
        ren.AddActor(_polydata_actor(polyd, Color='Aqua'))
        sagpln = midsagittal_plane(lmk[group==-1], lmk[group==0], lmk[group==1], weighting_func=None)
        plane = _plane_polydata(sagpln, np.array((lmk.min(axis=0),lmk.max(axis=0))).T.flatten())
        ren.AddActor(_polydata_actor(plane, Color='Aqua'))

        polyd = _points_polydata(_numpy_to_vtkpoints(lmk_nan))
        ren.AddActor(_polydata_actor(polyd, Color='IndianRed'))
        sagpln = midsagittal_plane(lmk_nan[group==-1], lmk_nan[group==0], lmk_nan[group==1], weighting_func=None)
        plane = _plane_polydata(sagpln, np.array((np.nanmin(lmk_nan, axis=0),np.nanmax(lmk_nan, axis=0))).T.flatten())
        ren.AddActor(_polydata_actor(plane, Color='IndianRed'))
        iren.Start()

    # '''
    # CONCLUSION:
    # difference is not significant
    # '''
    

    #########################################################################
    # THE EFFECT OF WEIGHTING
    #########################################################################
#     d1, landmarks_registered_weight, _ = iterative_procrustes(*landmarks, weighting_func=weighting_func, target_shape_index=0)

#     for i,(lmk, lmk_wgt) in enumerate(zip(landmarks_registered, landmarks_registered_weight)):
#         iren, ren = _render_window(f'd0={d0[i]}, d1={d1[i]}')        
#         polyd = _points_polydata(_numpy_to_vtkpoints(lmk))
#         ren.AddActor(_polydata_actor(polyd, Color='Aqua'))
#         sagpln = midsagittal_plane(lmk[group==-1], lmk[group==0], lmk[group==1], weighting_func=weighting_func)
#         plane = _plane_polydata(sagpln, np.array((lmk.min(axis=0),lmk.max(axis=0))).T.flatten())
#         ren.AddActor(_polydata_actor(plane, Color='Aqua'))
#         polyd = _points_polydata(_numpy_to_vtkpoints(lmk_wgt))
#         ren.AddActor(_polydata_actor(polyd, Color='IndianRed'))
#         sagpln = midsagittal_plane(lmk_wgt[group==-1], lmk_wgt[group==0], lmk_wgt[group==1], weighting_func=weighting_func)
#         plane = _plane_polydata(sagpln, np.array((np.nanmin(lmk_wgt, axis=0),np.nanmax(lmk_wgt, axis=0))).T.flatten())
#         ren.AddActor(_polydata_actor(plane, Color='IndianRed'))
#         iren.Start()
        
    # '''
    # CONCLUSION:
    # with weighting, some cases get worse and some better
    # seen in both number and figure
    # some differences are quite significant
    # '''
    
if __name__ == '__main__':
    prelim_test()