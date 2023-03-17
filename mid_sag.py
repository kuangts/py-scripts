import numpy as np
import scipy
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkPlaneSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch, vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor
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
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkImplicitDataSet
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkUnsignedCharArray, vtkFloatArray
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


error_range = (0.,30.)
num_bins = 100
def get_jet_color(normalized_values):
    x = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    r = [0.0, 0.0, 0.0, 1.0, 1.0, 0.5]
    g = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    b = [0.5, 1.0, 1.0, 0.0, 0.0, 0.0]
    rgb = np.asarray((
        np.interp(normalized_values, x, r),
        np.interp(normalized_values, x, g),
        np.interp(normalized_values, x, b),
    )).T
    return rgb


def polydata_to_numpy(polydata):
    numpy_nodes = vtk_to_numpy(polydata.GetPoints().GetData())
    numpy_faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1,4)[:,1:].astype(int)
    return encap(**{
        'nodes':numpy_nodes,
        'faces':numpy_faces
    })



class ChestVisualizer(vtk_basic.Visualizer):
    
    @staticmethod
    def decimate(v, f):
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(v,f)
        mesh_simplifier.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)
        vertices, faces, normals = mesh_simplifier.getMesh()
        return vertices, faces, normals


    def __init__(self, data):

        super().__init__()
        self.style.AddObserver('KeyPressEvent', self.key_press_event)

        self.lut = vtk.vtkLookupTable()
        self.lut.SetRange(-error_range[1], error_range[1]) # image intensity range
        # self.lut.SetValueRange(0.0, 1.0) # from black to white
        # self.lut.SetSaturationRange(0.0, 0.0) # no color saturation
        self.lut.SetRampToLinear()
        self.lut.Build()

        #### REMOVE DUPLICATE POITNS ####
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(data)
        cleaner.Update()
        port = cleaner.GetOutputPort()

        #### REDUCE THE NUMBER OF POINTS IF NECESSARY ####
        target_number_of_points = 10_000
        current_number_of_points = port.GetProducer().GetOutput().GetNumberOfPoints()
        if current_number_of_points > target_number_of_points:
            reducer = vtk.vtkQuadricDecimation() # also polydata algorithm
            reducer.SetInputConnection(port)
            target_reduction = 1 - target_number_of_points/current_number_of_points
            reducer.SetTargetReduction(target_reduction)
            reducer.Update()
            port = reducer.GetOutputPort()
        self.data_ready = port.GetProducer().GetOutput()
        self.d_clip = vtkImplicitPolyDataDistance()
        self.d_clip.SetInput(self.data_ready)

        #### INITIAL PLANE AND CAMERA ####
        self.plane = vtkPlane()
        self.mirror = MatrixTransform()
        self.bounds = port.GetProducer().GetOutput().GetBounds()
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        xdim, ydim, zdim = xmax-xmin, ymax-ymin, zmax-zmin
        xmid, ymid, zmid = xmin/2+xmax/2, ymin/2+ymax/2, zmin/2+zmax/2
        self.plane_widget = vtkImplicitPlaneWidget2()
        # self.plane_rep = vtkImplicitPlaneRepresentation()
        self.plane_widget.CreateDefaultRepresentation()
        self.plane_rep = self.plane_widget.GetImplicitPlaneRepresentation()
        self.plane_rep.SetWidgetBounds(xmin-xdim*.2,xmax+xdim*.2,ymin-ydim*.2,ymax+ydim*.2,zmin-zdim*.2,zmax+zdim*.2)
        self.plane_rep.SetOrigin(xmid, ymin-ydim*.1, zmid)
        self.plane_rep.SetNormal(1., 0., 0.)
        self.plane_rep.GetPlane(self.plane)
        self.plane_rep.SetDrawOutline(False)
        self.plane_widget.SetInteractor(self.iren)
        self.plane_widget.AddObserver('InteractionEvent', lambda *_:self.update_plane())
        self.plane_widget.AddObserver('EndInteractionEvent', lambda *_:self.plane_changed())
        self.plane_widget.AddObserver('StartInteractionEvent', lambda *_:self.plane_start())
        cam = self.ren.GetActiveCamera()
        cam.SetPosition(xmid, ymid-ydim*4, zmid)
        cam.SetFocalPoint(xmid, ymid, zmid)
        cam.SetViewUp(0., 0., 1.)
        self.plane_widget.On()

        #### WHOLE MODEL ####
        self.whole = Model(port, make_actor=False)

        #### WHOLE MODEL REFLECTED BY PLANE ####
        self.mirrored = Model(port, make_actor=False, transform=self.mirror)

        #### A PLANE CLIPS MODEL INTO HALVES ####
        self.clipper = vtkClipPolyData()
        self.clipper.SetClipFunction(self.plane)
        self.clipper.SetInputConnection(self.whole.outputport)
        self.clipper.GenerateClippedOutputOn()
        self.clipper.Update()
        half0_clipped = Model(self.clipper.GetOutputPort(0), color=colors.GetColor3d('red'))
        half1_clipped = Model(self.clipper.GetOutputPort(1), color=colors.GetColor3d('green'))
        
        self.add_actor(
            half0=half0_clipped.actor,
            half1=half1_clipped.actor,
        )
        #### HALF MODELS ####
        #### REMOVE DUPLICATE POINTS ####
        _cleaner = vtk.vtkCleanPolyData()
        _cleaner.SetInputConnection(self.clipper.GetOutputPort(0))
        _cleaner.Update()
        self.half0 = Model(_cleaner.GetOutputPort(), make_actor=False)

        _cleaner = vtk.vtkCleanPolyData()
        _cleaner.SetInputConnection(self.clipper.GetOutputPort(1))
        _cleaner.Update()
        self.half1m = Model(_cleaner.GetOutputPort(), transform=self.mirror)
        self.add_actor(
            half1m=self.half1m.actor,
        )


        self.cutoff_clipper = vtkClipPolyData()
        self.cutoff_clipper.SetInputData(self.half1m.output)
        # self.cutoff_clipper.SetClipFunction(self.d_clip)
        self.cutoff_clipper.GenerateClippedOutputOn()

        self.cutoff_clipper.SetValue(10)
        self.cutoff_clipper.GenerateClipScalarsOff()
        self.cutoff_clipper.InsideOutOn()
        self.cutoff_clipper.Update()
        self.half1m_good = Model(self.cutoff_clipper.GetOutputPort(0))
        self.half1m_bad = Model(self.cutoff_clipper.GetOutputPort(1))
        self.add_actor(
            half1m_good=self.half1m_good.actor,
            # half1m_bad=self.half1m_bad.actor,
        )
        # self.half1m_good.actor.GetMapper().SetScalarRange(error_range)
        # self.half1m_good.actor.GetMapper().SetLookupTable(self.lut)


        #### DISPLAY HALF MODELS ####

        self.ren_win.Render()
        self.update_plane()

        self.fig = plt.figure()
        self.ax = plt.gca()
        self.bar = self.ax.bar(np.linspace(0,1,num_bins), np.zeros((num_bins,)), width=.1)

        # https://matplotlib.org/stable/users/explain/event_handling.html

        def onclick(event):
            # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #     ('double' if event.dblclick else 'single', event.button,
            #     event.x, event.y, event.xdata, event.ydata))
            if event.button == 1 and event.dblclick:
                self.cutoff_clipper.SetValue(event.xdata)
                self.cutoff_clipper.Update()
                self.ren_win.Render()
                print(event.xdata)

        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
        plt.ion()
        plt.show()


    def key_press_event(self, obj, event):
        key = self.iren.GetKeySym()
        if key == 'k':
            self.register()

    def update_plane(self):
        self.plane_rep.GetPlane(self.plane)
        print(f'plane is updating {tic()}')
        origin = np.array(self.plane.GetOrigin())
        normal = np.array(self.plane.GetNormal())
        T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -origin.dot(normal) ]])
        self.mirror.update_matrix(T)
        # self.evaluate_distance()
        self.ren_win.Render()

    def reset_plane(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        xmid, ymid, zmid = xmin/2+xmax/2, ymin/2+ymax/2, zmin/2+zmax/2
        self.set_plane(origin=(xmid, ymid, zmid), normal=(1, 0, 0))

    def set_plane(self, origin, normal):
        self.plane_rep.SetOrigin(*origin)
        self.plane_rep.SetNormal(*normal)
        self.update_plane()


    def plane_changed(self):

        err = vtkFloatArray()
        err.SetName('Errors')
        self.d_clip.EvaluateFunction(self.half1m.output.GetPoints().GetData(), err)
        err_np = vtk_to_numpy(err)
        err_np_abs = np.abs(err_np)
        err_abs = vtkFloatArray()
        err_abs.DeepCopy(err_np_abs.ravel())
        # err_abs = numpy_to_vtk(err_np_abs.astype(np.float32), deep=True, array_type=err.GetDataType())
        # err_abs.SetNumberOfComponents(1)
        self.half1m.output.GetPointData().SetScalars(err_abs)
        self.half1m.output.GetPointData().SetActiveScalars('Errors')
        self.cutoff_clipper.Update()

        arr, edgs = np.histogram(err_np_abs, bins=num_bins)
        edgs = edgs[0:-1]/2 + edgs[1:]/2
        rgb = get_jet_color(edgs/self.cutoff_clipper.GetValue())
        wid = edgs[1]-edgs[0]
        if hasattr(self, 'bar'):
            for b,a,e,c in zip(self.bar, arr, edgs, rgb):
                b.set(x=e-.05, y=0, height=a, facecolor=c, width=wid)
            plt.xlim((edgs[0], edgs[-1]))
            plt.ylim((0, arr.max()))

        self.half1m.actor.VisibilityOff()
        self.half1m_good.actor.VisibilityOn()



    def plane_start(self):
        pass
        self.half1m.actor.VisibilityOn()
        self.half1m_good.actor.VisibilityOff()
        # add transparency to self.half1m.actor

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
        

        srcmm = self.half1m.output
        src = self.half1.output
        tar = self.half0.output
        reg_vtk = vtkPolyData()
        reg_vtk.DeepCopy(srcmm)

        srcm =  polydata_to_numpy(srcmm)
        src =  polydata_to_numpy(src)
        tar =  polydata_to_numpy(tar)
        
        reg = self.nicp(tar, srcm)

        if 'half1r' in self.actors:
            self.remove_actor('half1r')
            del self.half1r

        reg_vtk.GetPoints().SetData(numpy_to_vtk(reg.nodes))
        # self.half1r = Model(reg_vtk)
        # self.add_actor(half1r=self.half1r.actor)

        midpts = vtkPoints()
        midpts.SetData(numpy_to_vtk( src.nodes/2 + reg.nodes/2 ))
        origin, normal = [0.]*3, [0.]*3
        vtkPlane.ComputeBestFittingPlane(midpts, origin, normal)
        self.set_plane(origin, normal)
        self.plane_changed()


def midsagittal(stl_file_input):
    reader = vtkSTLReader()
    reader.SetFileName(stl_file_input)
    reader.Update()
    d = ChestVisualizer(reader.GetOutput())
    d.start()
    


if __name__ == '__main__':
    midsagittal(r'C:\data\midsagittal\skin_smooth_3mm.stl')
