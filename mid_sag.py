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
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkCommonDataModel import vtkPointSet
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkUnsignedCharArray
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from pycpd import DeformableRegistration
from argparse import Namespace as encap
import saikiran321


error_range = (0,30)
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

        #### REMOVE DUPLICATE POITNS ####
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(data)
        cleaner.Update()
        port = cleaner.GetOutputPort()

        #### REDUCE THE NUMBER OF POINTS IF NECESSARY ####
        target_number_of_points = 1_000
        current_number_of_points = port.GetProducer().GetOutput().GetNumberOfPoints()
        if current_number_of_points > target_number_of_points:
            reducer = vtk.vtkQuadricDecimation() # also polydata algorithm
            reducer.SetInputConnection(port)
            target_reduction = 1 - target_number_of_points/current_number_of_points
            reducer.SetTargetReduction(target_reduction)
            reducer.Update()
            port = reducer.GetOutputPort()

        srfc = polydata_to_numpy(port.GetProducer().GetOutput())
        self.tree = KDTree(srfc.nodes)

        #### INITIAL PLANE AND CAMERA ####
        self.plane = vtkPlane()
        self.mirror = MatrixTransform()
        xmin, xmax, ymin, ymax, zmin, zmax = port.GetProducer().GetOutput().GetBounds()
        xdim, ydim, zdim = xmax-xmin, ymax-ymin, zmax-zmin
        xmid, ymid, zmid = xmin/2+xmax/2, ymin/2+ymax/2, zmin/2+zmax/2
        self.plane_rep = vtkImplicitPlaneRepresentation()
        self.plane_rep.SetWidgetBounds(xmin-xdim*.2,xmax+xdim*.2,ymin-ydim*.2,ymax+ydim*.2,zmin-zdim*.2,zmax+zdim*.2)
        self.plane_rep.SetOrigin(xmid, ymin-ydim*.1, zmid)
        self.plane_rep.SetNormal(1., 0., 0.)
        self.plane_rep.SetDrawOutline(False)
        plane_widget = vtkImplicitPlaneWidget2()
        plane_widget.SetRepresentation(self.plane_rep)
        plane_widget.SetInteractor(self.iren)
        plane_widget.AddObserver('InteractionEvent', lambda *_:self.update_plane())
        cam = self.ren.GetActiveCamera()
        cam.SetPosition(xmid, ymid-ydim*4, zmid)
        cam.SetFocalPoint(xmid, ymid, zmid)
        cam.SetViewUp(0., 0., 1.)
        plane_widget.On()

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

        #### HALF MODELS ####
        #### REMOVE DUPLICATE POINTS ####
        _cleaner = vtk.vtkCleanPolyData()
        _cleaner.SetInputConnection(self.clipper.GetOutputPort(0))
        _cleaner.Update()
        self.half0 = Model(_cleaner.GetOutputPort(), color=colors.GetColor3d('red'))

        _cleaner = vtk.vtkCleanPolyData()
        _cleaner.SetInputConnection(self.clipper.GetOutputPort(1))
        _cleaner.Update()
        self.half1 = Model(_cleaner.GetOutputPort(), color=colors.GetColor3d('green'))
        
        self.half1m = Model(self.half1.outputport, transform=self.mirror)

        #### DISPLAY HALF MODELS ####
        self.add_actor(
            half0=self.half0.actor,
            half1=self.half1.actor,
            half1m=self.half1m.actor,
        )
        self.update_plane()
        self.ren_win.Render()
        # self.register()
        pass


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
        self.ren_win.Render()
        
        polyd = polydata_to_numpy(self.half1m.output)
        err, tar_nn = self.tree.query(polyd.nodes, k=1)

        arr, edgs = np.histogram(err, bins=100)
        edgs = edgs[0:-1]/2 + edgs[1:]/2

        shown = True
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()
            self.ax = plt.gca()
            self.bar = self.ax.bar(edgs, arr, width=.4)
            plt.show()
        for b,a,e in zip(self.bar,arr,edgs):
            b.set_height(a)
            b.set_xy((e-b.get_width(),0))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        err = (err-error_range[0]) / (error_range[1]-error_range[0])
        rgb = get_jet_color(err)
        c = numpy_to_vtk(np.round(rgb*255).astype(np.uint8), array_type=3) # unsigned char array
        c.SetNumberOfComponents(3)
        c.SetName('Colors')
        self.half1m.output.GetPointData().SetScalars(c)
        self.half1m.output.GetPointData().SetActiveScalars('Colors')
        c.Modified()




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
        self.plane_rep.SetOrigin(*origin)
        self.plane_rep.SetNormal(*normal)
        self.update_plane()

        self.ren_win.Render()


def midsagittal(stl_file_input):
    reader = vtkSTLReader()
    reader.SetFileName(stl_file_input)
    reader.Update()
    d = ChestVisualizer(reader.GetOutput())
    d.start()
    


if __name__ == '__main__':
    midsagittal(r'out.stl')
