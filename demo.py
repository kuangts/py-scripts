import sys
import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkPlaneSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch, vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleTerrain
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPlane, vtkIterativeClosestPointTransform
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkClipPolyData
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkTextActor,
    vtkPolyDataMapper,
    vtkCellPicker,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkInteractionWidgets import vtkImplicitPlaneRepresentation,vtkImplicitPlaneWidget2
import vtk
from vtk import vtkPolyData
from basic import Poly
from register import nicp
from scipy.spatial import KDTree
from time import perf_counter as tic
from numpy import all, any, eye, sum, mean, sort, unique, bincount, isin, exp, inf
from scipy.spatial import KDTree, distance_matrix
from numpy.linalg import svd, det, solve

colors = vtkNamedColors()

class Demo:
    def __init__(self):

        # create a rendering window and renderer
        ren = vtkRenderer()
        ren.SetBackground(colors.GetColor3d('PaleGoldenrod'))
        renwin = vtkRenderWindow()
        renwin.AddRenderer(ren)
        renwin.SetWindowName('Demo')

        # create a renderwindowinteractor
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renwin)
        style = vtkInteractorStyleTerrain()
        style.LatLongLinesOn()
        style.AddObserver('LeftButtonPressEvent', self.left_button_down_event)
        style.AddObserver('KeyPressEvent', self.key_press_event)
        iren.SetInteractorStyle(style)

        plane = vtkPlane()

        reader = vtkSTLReader()
        reader.SetFileName(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\manu-mand.stl')
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
        plane_rep.SetOrigin(xmid, ymin-ydim*.1, zmid)
        plane_rep.SetNormal(1., 0., 0.)
        plane_rep.SetDrawOutline(False)
        plane_widget = vtkImplicitPlaneWidget2()
        plane_widget.SetRepresentation(plane_rep)
        plane_widget.SetInteractor(iren)
        plane_widget.On()
        cam = ren.GetActiveCamera()
        cam.SetPosition(xmid, ymid-ydim*4, zmid)
        cam.SetFocalPoint(xmid, ymid, zmid)
        cam.SetViewUp(0., 0., 1.)


        clipper = vtkClipPolyData()
        clipper.SetClipFunction(plane)
        clipper.SetInputConnection(cleaner.GetOutputPort())
        clipper.GenerateClippedOutputOn()
        clipper.Update()

        self.half0 = vtk.vtkCleanPolyData()
        self.half0.SetInputConnection(clipper.GetOutputPort(0))
        self.half0.Update()

        self.half1 = vtk.vtkCleanPolyData()
        self.half1.SetInputConnection(clipper.GetOutputPort(1))
        self.half1.Update()

        # self.half1t = vtkTransformPolyDataFilter()
        # T = vtkMatrixToLinearTransform()
        # T.SetInput(vtkMatrix4x4())
        # T.Update()
        # self.half1t.SetTransform(T)
        # self.half1t.Update()

        plane_widget.AddObserver('InteractionEvent', self.update)

        ####################
        ###### DISPLAY #####
        ####################

        #### HALF0 ####
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self.half0.GetOutputPort())
        # mapper.SetColorModeToDirectScalars()
        # mapper.SetScalarRange(0.,10.)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('Red'))
        actor.PickableOff()
        ren.AddActor(actor)
        self.actor0 = actor

        #### HALF1 ####
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self.half1.GetOutputPort())
        # mapper.SetColorModeToDirectScalars()
        # mapper.SetScalarRange(0.,10.)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('Blue'))
        actor.PickableOff()
        ren.AddActor(actor)
        self.actor1 = actor

        # #### HALF1 t ####
        # mapper = vtkPolyDataMapper()
        # mapper.SetInputConnection(self.half1t.GetOutputPort())
        # actor = vtkActor()
        # actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(colors.GetColor3d('Yellow'))
        # ren.AddActor(actor)
        # self.actor1t = actor

        status = vtkTextActor()
        status.SetPosition2(10, 40)
        status.GetTextProperty().SetFontSize(16)
        status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        status.SetInput('AAAAA')
        ren.AddActor2D(status)

        self.widget = plane_widget
        self.iren = iren
        self.renwin = renwin
        self.ren = ren
        self.clipper = clipper
        self.plane_rep = plane_rep
        self.plane = plane 
        self.style = style


        iren.Initialize()
        renwin.Render()
        iren.Start()


    def left_button_down_event(self, obj, event):
        # # Get the location of the click (in window coordinates)
        # pos = obj.GetInteractor().GetEventPosition()
        # picker = vtkCellPicker()
        # picker.SetTolerance(0.0005)
        # # Pick from this location.
        # picker.Pick(pos[0], pos[1], 0, self.ren)
        # Forward events
        obj.OnLeftButtonDown()

    def update(self, obj, event):
        self.plane_rep.GetPlane(self.plane)
        self.clipper.Update()
        self.half0.Update()
        self.half1.Update()
        # self.half1t.Update()
        # self.renwin.Render()

    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(key)
        # if key != 'x': return
        # if isinstance(obj, vtkInteractorStyleTrackballCamera):
        #     self.style.SetCurrentStyleToTrackballActor()
        #     self.widget.Off()
        #     self.actor1.SetVisibility(False)

        #     #### HALF1 t ####
        #     pd = vtkPolyData()
        #     pd.DeepCopy(self.half1.GetOutput())
        #     mapper = vtkPolyDataMapper()
        #     mapper.SetInputData(pd)
        #     actor = vtkActor()
        #     actor.SetMapper(mapper)
        #     actor.GetProperty().SetColor(colors.GetColor3d('Yellow'))
        #     self.ren.AddActor(actor)
        #     self.actor1t = actor

        #     self.actor1t.SetVisibility(True)

        # elif isinstance(obj, vtkInteractorStyleTrackballActor):
        #     self.style.SetCurrentStyleToTrackballCamera()
        #     self.widget.On()
        #     self.actor1.SetVisibility(True)
        #     self.ren.RemoveActor(self.actor1t)
        #     del self.actor1t
            
        self.renwin.Render()

if __name__=='__main__':
    Demo()