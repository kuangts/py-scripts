#!/usr/bin/env python

import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkPlaneSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch, vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPlane
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkClipPolyData
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
from basic import Poly
from register import nicp
from scipy.spatial import KDTree
from time import perf_counter as tic

colors = vtkNamedColors()

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

def remove_duplicate(points, connectivity):
    if not points.size or not connectivity.size:
        return
    points = points.round(decimals=6)
    points, ind = np.unique(points, axis=0, return_inverse=True)
    connectivity = np.unique(ind[connectivity], axis=0)
    f_unique, ind = np.unique(connectivity, return_inverse=True)
    connectivity = ind.reshape(connectivity.shape)
    points = points[f_unique,:]
    return points, connectivity


class MouseInteractorStyle(vtkInteractorStyleSwitch):

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.moved = False
        self.should_rotate = True
        self.SetCurrentStyleToTrackballActor()
        self.style_actor = self.GetCurrentStyle()
        self.style_actor.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        self.style_actor.AddObserver('LeftButtonReleaseEvent', self.left_button_release_event)
        self.style_actor.AddObserver('RightButtonPressEvent', lambda o,e: None)
        self.style_actor.AddObserver('RightButtonReleaseEvent', lambda o,e: None)
        self.style_actor.AddObserver('MouseMoveEvent', self.mouse_move_event)
        self.SetCurrentStyleToTrackballCamera()
        self.style_camera = self.GetCurrentStyle()
        self.style_camera.AddObserver('KeyPressEvent', self.key_press_event)
        self.style_camera.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        self.style_camera.AddObserver('LeftButtonReleaseEvent', self.left_button_release_event)
        self.style_camera.AddObserver('MouseMoveEvent', self.mouse_move_event)
        self.style_camera.AddObserver('InteractionEvent', self.do_nothing)


    def do_nothing(self, obj, event):
        print(type)

    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        if key == 'r':
            self.visualizer.find_plane()
        if key.isdigit():
            self.visualizer.update_plane(key)

    def left_button_press_event(self, obj, event):
        self.moved = False
        self.should_rotate = True
        if self.GetCurrentStyle() == self.style_actor:
            pos = obj.GetInteractor().GetEventPosition()
            self.picker.Pick(pos[0], pos[1], 0, self.picker.ren)
            self.should_rotate = self.picker.GetActor() == self.visualizer.actor_plane
            if not self.should_rotate:
                return
        obj.OnLeftButtonDown()
                
    def mouse_move_event(self, obj, event):
        self.moved = True
        if self.GetCurrentStyle() == self.style_actor and self.should_rotate:
            obj.OnMouseMove()
            self.visualizer.update_plane()
        elif self.GetCurrentStyle() == self.style_camera:
            obj.OnMouseMove()

    def left_button_release_event(self, obj, event):
        if self.GetCurrentStyle() == self.style_actor and self.should_rotate:
            obj.OnLeftButtonUp()
        elif self.GetCurrentStyle() == self.style_camera:
            obj.OnLeftButtonUp()
        if not self.moved:
            self.clicked(obj, event)

    def clicked(self, obj, event):
        pos = obj.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, self.picker.ren)
        if self.picker.GetActor() == self.visualizer.actor_plane:
            self.visualizer.select_plane()
        else:
            self.visualizer.deselect_plane()



class Visualizer:
    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        if key == 'r':
            self.find_plane()
        # if key.isdigit():
        #     self.update_plane(key)

    def find_plane(self):
        tar_np = polydata_to_numpy(self.half0.GetOutput())
        tar = Poly(V=tar_np['nodes'],F=tar_np['faces'])
        src_np = polydata_to_numpy(self.half1.GetOutput())

        v_src = src_np['nodes']
        v_src = np.hstack((v_src, np.ones((v_src.shape[0],1)))) @ vtkMatrix4x4_to_numpy(self._T).T
        src = Poly(V=v_src[:,:3],F=src_np['faces'])

        reg = nicp(src, tar)
        reg_np = dict(nodes=reg.V, faces=reg.F)

        self.nicp_result = numpy_to_polydata(**reg_np)
        self.nicp_mapper.SetInputData(self.nicp_result)
        self.nicp_mapper.Update()
        tar_tree = KDTree(tar.V)
        d, nn = tar_tree.query(reg_np['nodes'], k=1)
        # keep = d<10
        # points = src_np['nodes'][keep,:]/2 + tar_np['nodes'][nn[keep],:]/2
        points = src_np['nodes']/2 + tar_np['nodes'][nn,:]/2


#         colors = vtk.vtkUnsignedCharArray()
#         colors.SetNumberOfComponents(1)
#         colors.SetNumberOfValues(2)
# for c in range(Cellarray):
#     Colors.InsertNextTuple3(0, 255, 0)
# coneSource.GetOutput().GetCellData().SetScalars(d)

        cen = points.mean(axis=0, keepdims=True)
        points = points - cen
        _, _, W = np.linalg.svd(points.T@points)
        normal = W[-1].flatten()        
        normal = normal/np.sum(normal**2)**.5
        if normal[0]<0: normal *= -1
        self.set_plane(cen.flat, normal)
        self.update_T()

    def set_plane(self, origin, normal):
        self.plane_rep.SetOrigin(*origin)
        self.plane_rep.SetNormal(*normal)

    def record_plane(self, key):
        self._planes[key] = dict(
            origin=self.plane_rep.GetOrigin(),
            normal=self.plane_rep.GetNormal(),
        )
        
    def update_T(self):
        self.plane_rep.GetPlane(self.plane)
        print(f'plane is updating {tic()}')
        origin = np.array(self.plane.GetOrigin())
        normal = np.array(self.plane.GetNormal())
        T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -origin.dot(normal) ]])
        self._T.DeepCopy(T.ravel())
        self.half0.Update()
        self.half1.Update()
        self.half1m.Update()


    def __init__(self, skin_path, cage_path=None, initial_plane=None):

        # create a rendering window and renderer
        ren = vtkRenderer()
        ren.SetBackground(colors.GetColor3d('PaleGoldenrod'))
        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetWindowName('InteractorStyleTrackballCamera')

        # create a renderwindowinteractor
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        style = vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
        style.AddObserver('KeyPressEvent', self.key_press_event)

        self._T = vtkMatrix4x4()
        self.plane = vtkPlane()        
        self._planes = {}

        reader = vtkSTLReader()
        reader.SetFileName(skin_path)
        reader.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(reader.GetOutputPort())
        cleaner.Update()

        self.decimator = vtk.vtkDecimatePro()
        self.decimator.SetTargetReduction(.9)
        self.decimator.SetInputConnection(cleaner.GetOutputPort())
        self.decimator.Update()

        #### INITIAL PLANE AND CAMERA ####
        xmin, xmax, ymin, ymax, zmin, zmax = self.decimator.GetOutput().GetBounds()
        xdim, ydim, zdim = xmax-xmin, ymax-ymin, zmax-zmin
        xmid, ymid, zmid = xmin/2+xmax/2, ymin/2+ymax/2, zmin/2+zmax/2
        self.plane_rep = vtkImplicitPlaneRepresentation()
        self.plane_rep.SetWidgetBounds(xmin-xdim*.2,xmax+xdim*.2,ymin-ydim*.2,ymax+ydim*.2,zmin-zdim*.2,zmax+zdim*.2)
        self.plane_rep.SetOrigin(xmid, ymin-ydim*.1, zmid)
        self.plane_rep.SetNormal(1., 0., 0.)
        self.plane_rep.SetDrawOutline(False)
        plane_widget = vtkImplicitPlaneWidget2()
        plane_widget.SetRepresentation(self.plane_rep)
        plane_widget.SetInteractor(iren)
        plane_widget.On()
        plane_widget.AddObserver('InteractionEvent', lambda *_:self.update_T())
        cam = ren.GetActiveCamera()
        cam.SetPosition(xmid, ymid-ydim*4, zmid)
        cam.SetFocalPoint(xmid, ymid, zmid)
        cam.SetViewUp(0., 0., 1.)

        #### REFLECTION BY PLANE ####
        self.mirror = vtkMatrixToLinearTransform()
        self.mirror.SetInput(self._T)
        self.mirror.Update()

        self.clipper = vtkClipPolyData()
        self.clipper.SetClipFunction(self.plane)
        self.clipper.SetInputConnection(self.decimator.GetOutputPort())
        self.clipper.GenerateClippedOutputOn()

        self.half0 = vtk.vtkCleanPolyData()
        self.half0.SetInputConnection(self.clipper.GetOutputPort(0))
        self.half0.Update()

        self.half1 = vtk.vtkCleanPolyData()
        self.half1.SetInputConnection(self.clipper.GetOutputPort(1))
        self.half1.Update()

        self.half1m = vtkTransformPolyDataFilter()
        self.half1m.SetTransform(self.mirror)
        self.half1m.SetInputConnection(self.half1.GetOutputPort())
        self.half1m.Update()

        self.update_T()

        # #### GISMO ####
        # axes = vtkAxesActor()
        # axes.SetShaftTypeToCylinder()
        # axes.SetTotalLength(1.0, 1.0, 1.0)
        # axes.SetCylinderRadius(0.5 * axes.GetCylinderRadius())
        # axes.SetConeRadius(1.025 * axes.GetConeRadius())
        # axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
        # gyro = vtkOrientationMarkerWidget()
        # gyro.SetOrientationMarker(axes)
        # gyro.SetInteractor(iren)
        # gyro.EnabledOn()
        # gyro.InteractiveOn()
        # gyro.SetViewport(0, 0, 0.1, 0.1)

        # #### MAIN MODEL ####
        # mapper = vtkPolyDataMapper()
        # mapper.SetInputConnection(reader.GetOutputPort())
        # mapper.SetColorModeToDirectScalars()
        # mapper.SetScalarRange(0.,10.)
        # actor_model = vtkActor()
        # actor_model.SetMapper(mapper)
        # actor_model.GetProperty().SetColor(colors.GetColor3d('Chartreuse'))
        # actor_model.GetProperty().SetOpacity(0.)
        # ren.AddActor(actor_model)

        # mirror = vtkTransformPolyDataFilter()
        # mirror.SetTransform(tfm)
        # mirror.SetInputConnection(reader.GetOutputPort())
        # mirror.Update()
        # mapper_model_mirrored = vtkPolyDataMapper()
        # mapper_model_mirrored.SetInputConnection(mirror.GetOutputPort())
        # actor_model_mirrored = vtkActor()
        # actor_model_mirrored.SetMapper(mapper_model_mirrored)
        # actor_model_mirrored.GetProperty().SetColor(colors.GetColor3d('Yellow'))
        # actor_model_mirrored.GetProperty().SetOpacity(0)
        # ren.AddActor(actor_model_mirrored)

        #### MIRRORED HALF MODEL ####
        mapper_half_mirrored = vtkPolyDataMapper()
        mapper_half_mirrored.SetInputConnection(self.half1m.GetOutputPort())
        actor_half_mirrored = vtkActor()
        actor_half_mirrored.SetMapper(mapper_half_mirrored)
        actor_half_mirrored.GetProperty().SetColor(colors.GetColor3d('Yellow'))
        ren.AddActor(actor_half_mirrored)

        #### NICP MODEL ####
        # updateT = vtk.vtkProgrammableFilter()
        # updateT.SetExecuteMethod(self.update_T)
        # updateT.Update()
        self.nicp_result = vtkPolyData()
        self.nicp_result.DeepCopy(self.half1m.GetOutput())
        self.nicp_mapper = vtkPolyDataMapper()
        self.nicp_mapper.SetInputData(self.nicp_result)
        self.nicp_mapper.Update()
        actor_incp = vtkActor()
        actor_incp.SetMapper(self.nicp_mapper)
        actor_incp.GetProperty().SetColor(colors.GetColor3d('Green'))
        ren.AddActor(actor_incp)

        #### HALF0 ####
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self.half0.GetOutputPort())
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarRange(0.,10.)
        actor_left = vtkActor()
        actor_left.SetMapper(mapper)
        actor_left.GetProperty().SetColor(colors.GetColor3d('Red'))
        ren.AddActor(actor_left)

        #### HALF1 ####
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self.half1.GetOutputPort())
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarRange(0.,10.)
        actor_right = vtkActor()
        actor_right.SetMapper(mapper)
        actor_right.GetProperty().SetColor(colors.GetColor3d('Blue'))
        ren.AddActor(actor_right)

        iren.Initialize()
        renWin.Render()
        iren.Start()


if __name__ == '__main__':

    Visualizer(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\p4\skin_smooth_3mm.stl')

