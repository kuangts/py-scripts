import numpy as np
import scipy
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkPlaneSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch, vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkCommonCore import vtkPoints, vtkScalarsToColors
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPlane, vtkIterativeClosestPointTransform
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
from numpy import all, any, eye, sum, mean, sort, unique, bincount, isin, exp, inf
from scipy.spatial import KDTree, distance_matrix
from numpy.linalg import svd, det, solve

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import sys, os, glob, json
from collections import namedtuple
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkImplicitPolyDataDistance
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkImplicitDataSet, vtkImplicitWindowFunction 
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkUnsignedCharArray, vtkFloatArray
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkTransformFilter, vtkDistancePolyDataFilter
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
from vtkmodules.vtkFiltersPython import vtkPythonAlgorithm
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
import numpy as np
from scipy.spatial  import KDTree
import vtk_basic
from vtk_basic import MatrixTransform, Model, colors
from copy import deepcopy
# import pyfqmr


from functools import partial
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# from pycpd import DeformableRegistration
from argparse import Namespace as encap
import saikiran321


default_error_range = (-20.,20.)
target_number_of_points = 5_000

num_bins = 100
# def get_jet_color(normalized_values):
#     x = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
#     r = [0.0, 0.0, 0.0, 1.0, 1.0, 0.5]
#     g = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
#     b = [0.5, 1.0, 1.0, 0.0, 0.0, 0.0]
#     rgb = np.asarray((
#         np.interp(normalized_values, x, r),
#         np.interp(normalized_values, x, g),
#         np.interp(normalized_values, x, b),
#     )).T
#     return rgb


def polydata_to_numpy(polydata):
    numpy_nodes = vtk_to_numpy(polydata.GetPoints().GetData())
    numpy_faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1,4)[:,1:].astype(int)
    return encap(**{
        'nodes':numpy_nodes,
        'faces':numpy_faces
    })

class MirrorTransform(vtkMatrixToLinearTransform):
    
    def __init__(self, mirror_plane=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mirror_plane is None:
            mirror_plane = vtkPlane()
        self.plane = mirror_plane
        self.SetInput(vtkMatrix4x4())

    # def update_matrix(self):
    #     origin = np.array(self.plane.GetOrigin())
    #     normal = np.array(self.plane.GetNormal())
    #     T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -origin.dot(normal) ]])
    #     self.GetInput().DeepCopy(T.ravel())

    # def Initialize(self, vtkself):
    #         vtkself.SetNumberOfInputPorts(1)
    #         vtkself.SetNumberOfOutputPorts(1)

    # def FillInputPortInformation(self, vtkself, port, info):
    #     info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
    #     return 1

    # def FillOutputPortInformation(self, vtkself, port, info):
    #     info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkDataSet")
    #     return 1
    
    def ProcessRequest(self, *args, **kwargs):
        origin = np.array(self.plane.GetOrigin())
        normal = np.array(self.plane.GetNormal())
        T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -origin.dot(normal) ]])
        self.GetInput().DeepCopy(T.ravel())
        super().ProcessRequest(*args, **kwargs)


class obj(object):
    pass

class ChestVisualizer(vtk_basic.Visualizer):
    
    @staticmethod
    def decimate(v, f):
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(v,f)
        mesh_simplifier.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)
        vertices, faces, normals = mesh_simplifier.getMesh()
        return vertices, faces, normals

    def build_color(self, ran):
        if not hasattr(self, 'lut'):
            self.lut = vtk.vtkColorTransferFunction()
        r = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5]
        g = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        b = [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        rgb = np.vstack((r,g,b)).T
        rgb = np.vstack((rgb[::-1,:], rgb)).ravel()
        self.lut.BuildFunctionFromTable(*ran, len(r), rgb)
        self.lut.Modified()

    def __init__(self, data):

        super().__init__()
        self.style.AddObserver('KeyPressEvent', self.key_press_event)
        self.build_color(default_error_range)

        #### REMOVE DUPLICATE POITNS ####
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(data)
        cleaner.Update()
        self.dataport = cleaner.GetOutputPort()

        #### REDUCE THE NUMBER OF POINTS IF NECESSARY ####
        current_number_of_points = self.dataport.GetProducer().GetOutput().GetNumberOfPoints()
        if current_number_of_points > target_number_of_points:
            reducer = vtk.vtkQuadricDecimation() # also polydata algorithm
            reducer.SetInputConnection(self.dataport)
            target_reduction = 1 - target_number_of_points/current_number_of_points
            reducer.SetTargetReduction(target_reduction)
            reducer.Update()
            self.dataport = reducer.GetOutputPort()

        self.data_ready = self.dataport.GetProducer().GetOutput()

        self.distance_function = vtkImplicitPolyDataDistance()
        self.distance_function.SetInput(self.data_ready)

        self.distance_clip_function = vtkImplicitWindowFunction()
        self.distance_clip_function.SetImplicitFunction(self.distance_function)
        self.distance_clip_function.SetWindowRange(default_error_range)

        self.distance_clip = vtkClipPolyData()
        self.distance_clip.SetClipFunction(self.distance_clip_function)
        self.distance_clip.GenerateClipScalarsOn()

        #### INITIAL PLANE AND CAMERA ####
        self.mirror = MatrixTransform()
        self.bounds = self.data_ready.GetBounds()
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        xdim, ydim, zdim = xmax-xmin, ymax-ymin, zmax-zmin
        xmid, ymid, zmid = xmin/2+xmax/2, ymin/2+ymax/2, zmin/2+zmax/2
        self.plane_widget = vtkImplicitPlaneWidget2()
        self.plane_widget.CreateDefaultRepresentation()
        self.plane_rep = self.plane_widget.GetImplicitPlaneRepresentation()
        self.plane = self.plane_rep.GetUnderlyingPlane()
        self.plane_rep.SetWidgetBounds(xmin-xdim*.2,xmax+xdim*.2,ymin-ydim*.2,ymax+ydim*.2,zmin-zdim*.2,zmax+zdim*.2)
        self.plane_rep.SetOrigin(xmid, ymin-ydim*.1, zmid)
        self.plane_rep.SetNormal(1., 0., 0.)
        self.plane_rep.SetDrawOutline(False)
        self.plane.AddObserver('ModifiedEvent', lambda *_: self.plane_modified())
        self.plane_widget.SetInteractor(self.iren)
        self.plane_widget.AddObserver('EndInteractionEvent', lambda *_:self.plane_end())
        cam = self.ren.GetActiveCamera()
        cam.SetPosition(xmid, ymid-ydim*4, zmid)
        cam.SetFocalPoint(xmid, ymid, zmid)
        cam.SetViewUp(0., 0., 1.)
        self.plane_widget.On()
        self.plane_modified()

        #### WHOLE MODEL ####
        self.whole = obj()
        self.whole.mapper = vtkPolyDataMapper()
        self.whole.mapper.SetInputData(self.data_ready)
        self.whole.data = self.data_ready
        self.whole.actor = vtkActor()
        self.whole.actor.SetMapper(self.whole.mapper)
        self.whole.actor.GetProperty().SetColor(.5,.5,.5)
        self.ren.AddActor(self.whole.actor)

        self.whole_mirrored = vtkTransformPolyDataFilter()
        self.whole_mirrored.SetTransform(self.mirror)
        self.whole_mirrored.SetInputData(self.data_ready)
        self.distance_clip.SetInputConnection(self.whole_mirrored.GetOutputPort())

        #### A PLANE CLIPS MODEL INTO HALVES ####
        self.plane_clipper = vtkClipPolyData()
        self.plane_clipper.SetClipFunction(self.plane)
        self.plane_clipper.GenerateClippedOutputOn()
        self.plane_clipper.SetInputConnection(self.whole_mirrored.GetOutputPort())
        _cleaner = vtk.vtkCleanPolyData()
        _cleaner.SetInputConnection(self.plane_clipper.GetOutputPort(0))
        self.half = obj()
        self.half.port = _cleaner.GetOutputPort()
        self.half.data = _cleaner.GetOutput()
        self.half.mapper = vtkPolyDataMapper()
        self.half.mapper.SetInputConnection(self.half.port)
        self.half.mapper.SetLookupTable(self.lut)
        self.half.actor = vtkActor()
        self.half.actor.SetMapper(self.half.mapper)
        self.ren.AddActor(self.half.actor)

        self.whole.mapper.Update()
        self.half.mapper.Update()
        self.update_distance()
        self.ren_win.Render()

        self.fig = plt.figure()
        self.ax = plt.gca()
        self.bar = self.ax.bar(range(num_bins), np.zeros((num_bins,)), width=.1)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.clicked_on_bar)
        cid = self.fig.canvas.mpl_connect('button_release_event', self.released_on_bar)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)
        plt.ion()
        plt.show()


    def clip(self, ran=None):
        if ran is None:
            self.whole_mirrored.SetInputData(self.data_ready)            
        else:
            self.distance_clip_function.SetWindowRange(ran)
            self.distance_clip.Update()
            transform = vtkTransformPolyDataFilter()
            transform.SetTransform(self.mirror)
            transform.SetInputConnection(self.distance_clip.GetOutputPort())
            transform.Update()
            polyd = vtkPolyData()
            polyd.DeepCopy(transform.GetOutput())
            self.whole_mirrored.SetInputData(polyd)         
        self.whole_mirrored.Update()
        
        self.half.mapper.Update()
        self.update_distance()
        self.ren_win.Render()
        


    def clicked_on_bar(self, event):
        # https://matplotlib.org/stable/users/explain/event_handling.html

        if event.button == 1:
            if event.dblclick:
                if hasattr(self, 'ran'):
                    del self.ran
                self.clip()
                self.ren_win.Render()
                return
            
            if not event.xdata:
                return
            self.ran = [event.xdata, event.xdata]
            if not hasattr(self, 'lines'):
                *_, ymin, ymax = self.ax.axis()
                self.lines = [
                    plt.axvline(event.xdata, ymin=ymin, ymax=ymax),
                    plt.axvline(event.xdata, ymin=ymin, ymax=ymax),
                ]
            else:
                for l in self.lines:
                    l.set_xdata([event.xdata, event.xdata])

    def motion_notify_event(self, event):
        if hasattr(self, 'ran') and event.xdata:
            self.lines[1].set_xdata([event.xdata, event.xdata])


    def released_on_bar(self, event):
        if event.button == 1:
            if event.xdata and hasattr(self, 'ran') and abs(event.xdata - self.ran[0]) > 5:
                self.ran[1] = event.xdata
                self.ran.sort()
                self.clip(self.ran)


            if hasattr(self, 'ran'):
                del self.ran

            if hasattr(self, 'lines'):
                self.ax.lines.remove(self.lines[0])
                self.ax.lines.remove(self.lines[1])
                del self.lines

            self.ren_win.Render()


    def key_press_event(self, obj, event):
        key = self.iren.GetKeySym()
        if not hasattr(self, 'command'):
            self.command = ''
        if key == 'k':
            self.command += key
            print(self.command)
            self.register()


    def plane_modified(self):
        # callback for plane modified
        # can use to refresh self.plane -> self.mirror -> pipeline
        print(f'plane is modified {tic()}')
        origin = np.array(self.plane.GetOrigin())
        normal = np.array(self.plane.GetNormal())
        T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -origin.dot(normal) ]])
        self.mirror.update_matrix(T)


    def set_plane(self, origin, normal):
        # update plane widget manually
        # then refresh
        self.plane_rep.SetOrigin(*origin)
        self.plane_rep.SetNormal(*normal)
        self.half.mapper.Update()
        self.update_distance()
        self.ren_win.Render()
        # for a in self.actors.values():
        #     a.GetMapper().Update()


    ## planes updates (involving color/error/clipping)
    def reset_plane(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        xmid, ymid, zmid = xmin/2+xmax/2, ymin/2+ymax/2, zmin/2+zmax/2
        self.set_plane(origin=(xmid, ymid, zmid), normal=(1, 0, 0))


    def plane_end(self):
        self.update_distance()
        self.half.mapper.Update()


    def update_distance(self):
        err = vtkFloatArray()
        self.distance_function.EvaluateFunction(self.half.data.GetPoints().GetData(), err)
        self.half.data.GetPointData().SetScalars(err)
        self.half.mapper.Update()
        self.ren_win.Render()

        arr, edgs = np.histogram(vtk_to_numpy(err), bins=num_bins)
        edgs = edgs[0:-1]/2 + edgs[1:]/2
        wid = edgs[1]-edgs[0]
        rgb = np.empty(edgs.size*4, "uint8")
        edg_arr = numpy_to_vtk(edgs, deep=0, array_type=10)
        self.lut.MapScalarsThroughTable(edg_arr, rgb)
        rgb = rgb.reshape(-1,4)/255
        if hasattr(self, 'bar'):
            self.ax.set_xlim(edgs[0], edgs[-1])
            self.ax.set_ylim(0, arr.max())
            for b,a,e,c in zip(self.bar, arr, edgs, rgb):
                b.set(x=e-wid/2, y=0, height=a, facecolor=c, width=wid)



    @staticmethod
    def cpd(tar, src):
        reg = DeformableRegistration(X=tar.nodes, Y=src.nodes)
        src_reg = deepcopy(src)
        src_reg.nodes, *_ = reg.register()
        return src_reg


    @staticmethod
    def nicp(tar, src):
        src_reg = deepcopy(src)
        reg = nicp(
            dict(V=src_reg.nodes,F=src_reg.faces),
            dict(V=tar.nodes,F=tar.faces))
        src_reg.nodes = reg.V
        return src_reg

    @staticmethod
    def saikiran321_nonrigidIcp(tar, src):
        src_reg = deepcopy(src)
        src_reg = saikiran321.nonrigidIcp(
            encap(vertices=src.nodes, triangles=src.faces),
            encap(vertices=tar.nodes, triangles=tar.faces),
        )
        src_reg.nodes = src_reg.vertices
        src_reg.faces = src_reg.triangles
        
        return src_reg


    def register(self):
        
        halfmm = Model(self.half.data, transform=self.mirror, show=True)

        srcm = self.half.data
        src = halfmm.output
        tar = self.whole.data
        reg_vtk = vtkPolyData()
        reg_vtk.DeepCopy(srcm)

        srcm =  polydata_to_numpy(srcm)
        src =  polydata_to_numpy(src)
        tar =  polydata_to_numpy(tar)
        
        reg = self.nicp(tar, srcm)

        reg_vtk.GetPoints().SetData(numpy_to_vtk(reg.nodes))

        midpts = vtkPoints()
        midpts.SetData(numpy_to_vtk( src.nodes/2 + reg.nodes/2 ))
        origin, normal = [0.]*3, [0.]*3
        vtkPlane.ComputeBestFittingPlane(midpts, origin, normal)
        self.set_plane(origin, normal)


def midsagittal(stl_file_input):
    reader = vtkSTLReader()
    reader.SetFileName(stl_file_input)
    reader.Update()
    d = ChestVisualizer(reader.GetOutput())
    d.start()
    


if __name__ == '__main__':
    midsagittal(r'out.stl')
