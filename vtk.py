#!/usr/bin/env python

# noinspection PyUnresolvedReferences
from vtkmodules.all import *
import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkPlaneSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtk import vtkMatrix4x4
from time import perf_counter as tic

skin_path = r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\p4\skin_smooth_3mm.stl'

def vtkMatrix4x4_to_numpy(mat4x4):
    t = np.eye(4)
    mat4x4.DeepCopy(t.ravel(), mat4x4)        
    return t

def numpy_to_vtkMatrix4x4(np4x4):
    mat = vtkMatrix4x4()
    mat.DeepCopy(np4x4.ravel())        
    return mat

class Visualizer:
    def get_plane(self):
        pass
    
    def mouse_moved(self):
        self.style.GetCurrentStyle().OnMouseMove()
        self.update_plane()

    def update_plane(self, *args):
        if not isinstance(self.style.GetCurrentStyle(), vtkInteractorStyleTrackballActor):
            return
        
        print(f'called {tic()}')
        T = vtkMatrix4x4_to_numpy(self.actor_plane.GetMatrix())
        center = np.array([100.,0.,0.])
        center = (T @ np.asarray([[*center, 1]]).T).flat[:-1]
        normal = np.array([1.,0.,0.])
        normal = (T @ np.asarray([[*normal,0.]]).T).flat[:-1]
        T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -center.dot(normal) ]])
        self._T.DeepCopy(T.ravel())
        self.mapper_model_mirrored.Update()

    def __init__(self, skin_path, cage_path=None, init_transform=vtkMatrix4x4()):

        colors = vtkNamedColors()
        reader = vtkSTLReader()
        reader.SetFileName(skin_path)
        self._T = vtkMatrix4x4()

        # create a rendering window and renderer
        ren = vtkRenderer()
        ren.SetBackground(colors.GetColor3d('PaleGoldenrod'))
        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetWindowName('InteractorStyleTrackballCamera')

        # create a renderwindowinteractor
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        style = vtkInteractorStyleSwitch()
        style.SetCurrentStyleToTrackballCamera()
        iren.SetInteractorStyle(style)

        #### GISMO ####
        axes = vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetCylinderRadius(0.5 * axes.GetCylinderRadius())
        axes.SetConeRadius(1.025 * axes.GetConeRadius())
        axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
        gyro = vtkOrientationMarkerWidget()
        gyro.SetOrientationMarker(axes)
        gyro.SetInteractor(iren)
        gyro.EnabledOn()
        gyro.InteractiveOn()
        gyro.SetViewport(0, 0, 0.1, 0.1)

        #### MODEL ####
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor_model = vtkActor()
        actor_model.SetMapper(mapper)
        actor_model.GetProperty().SetColor(colors.GetColor3d('Chartreuse'))
        actor_model.PickableOff()
        ren.AddActor(actor_model)

        #### MIRRORED MODEL ####
        transform = vtkMatrixToLinearTransform()
        transform.SetInput(self._T)
        transform.Update()
        transformer = vtkTransformPolyDataFilter()
        transformer.SetTransform(transform)
        transformer.SetInputConnection(reader.GetOutputPort())
        mapper_model_mirrored = vtkPolyDataMapper()
        mapper_model_mirrored.SetInputConnection(transformer.GetOutputPort(0))
        actor_model_mirrored = vtkActor()
        actor_model_mirrored.SetMapper(mapper_model_mirrored)
        actor_model_mirrored.GetProperty().SetColor(colors.GetColor3d('RoyalBLue'))
        actor_model_mirrored.PickableOff()
        ren.AddActor(actor_model_mirrored)

        #### PLANE ####
        plane_source = vtkPlaneSource()
        plane_source.SetOrigin(100.0, 0.0, 0.0)
        plane_source.SetPoint1(100.0, 100.0, 0.0)
        plane_source.SetPoint2(100.0, 0.0, 100.0)
        plane_source.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(plane_source.GetOutputPort())
        actor_plane = vtkActor()
        actor_plane.SetMapper(mapper)
        actor_plane.GetProperty().SetLineWidth(40)
        actor_plane.GetProperty().SetColor(colors.GetColor3d('Chartreuse'))
        ren.AddActor(actor_plane)

        self.ren = ren
        self.style = style
        self.iren = iren
        self.actor_model = actor_model
        self.plane_source = plane_source
        self.actor_plane = actor_plane
        self.transform = transform
        self.transformer = transformer
        self.actor_model_mirrored = actor_model_mirrored
        self.mapper_model_mirrored = mapper_model_mirrored
        # self.iren.SetStillUpdateRate(30.)
        # self.iren.SetDesiredUpdateRate(30.)

        iren.AddObserver('MouseMoveEvent', self.update_plane) # some events don't work
        iren.Initialize()
        renWin.Render()
        iren.Start()



# class ktsInteractorStyleSwitch(vtkInteractorStyleSwitch):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.SetCurrentStyleToTrackballCamera()

#     def SetCurrentStyleToJoystickActor(self):
#         self.SetCurrentStyleToTrackballActor()

#     def SetCurrentStyleToJoystickCamera(self):
#         self.SetCurrentStyleToTrackballCamera()

#     def OnKeyPress(self):
#         if hasattr(self, 'interactor'):
#             key = self.GetInteractor().GetKeySym()
#             print(key)
#             if key == 'x':
#                 if isinstance(self.GetCurrentStyle(), vtkInteractorStyleTrackballActor):
#                     self.SetCurrentStyleToTrackballCamera()
#                 elif isinstance(self.GetCurrentStyle(), vtkInteractorStyleTrackballCamera):
#                     self.SetCurrentStyleToTrackballActor()

#     def OnKeyRelease(self):
#         pass


if __name__ == '__main__':

    Visualizer(skin_path=skin_path)