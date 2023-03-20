from copy import deepcopy

import numpy as np
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkCommonCore import vtkPoints, vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkPlane, vtkImplicitWindowFunction
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkClipPolyData, vtkCleanPolyData, vtkImplicitPolyDataDistance, vtkQuadricDecimation
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkImplicitPlaneRepresentation, vtkImplicitPlaneWidget2
from vtkmodules.vtkFiltersPython import vtkPythonAlgorithm
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from register import nicp



default_error_range = (-20.,20.)
target_number_of_points = 5_000
histo_num_bins = 100


class encap(object):
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)


def polydata_to_numpy(polydata):
    numpy_nodes = vtk_to_numpy(polydata.GetPoints().GetData())
    numpy_faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1,4)[:,1:].astype(int)
    result = encap(nodes=numpy_nodes, faces=numpy_faces)
    return result


class ChestVisualizer:
    
    def build_color(self, ran):
        if not hasattr(self, 'lut'):
            self.lut = vtkColorTransferFunction()
        r = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5]
        g = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        b = [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        rgb = np.vstack((r,g,b)).T
        rgb = np.vstack((rgb[::-1,:], rgb)).ravel()
        self.lut.BuildFunctionFromTable(*ran, len(r), rgb)
        self.lut.Modified()

    def __init__(self, data):

        # create window 
        renderer = vtkRenderer()
        renderer.SetBackground(.67, .93, .93)
        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(renderer)
        ren_win.SetSize(960, 640)
        ren_win.SetWindowName('')
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        style = vtkInteractorStyleTrackballCamera()
        style.SetDefaultRenderer(renderer)
        iren.SetInteractorStyle(style)

        self.ren = renderer
        self.ren_win = ren_win
        self.iren = iren
        self.style = style

        self.style.AddObserver('KeyPressEvent', self.key_press_event)
        self.build_color(default_error_range)

        #### REMOVE DUPLICATE POITNS ####
        cleaner = vtkCleanPolyData()
        cleaner.SetInputData(data)
        cleaner.Update()
        self.dataport = cleaner.GetOutputPort()

        #### REDUCE THE NUMBER OF POINTS IF NECESSARY ####
        current_number_of_points = self.dataport.GetProducer().GetOutput().GetNumberOfPoints()
        if current_number_of_points > target_number_of_points:
            reducer = vtkQuadricDecimation() # also polydata algorithm
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
        self.mirror = vtkMatrixToLinearTransform()
        self.mirror.SetInput(vtkMatrix4x4())
        self.mirror.Update()
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
        self.whole = encap()
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
        _cleaner = vtkCleanPolyData()
        _cleaner.SetInputConnection(self.plane_clipper.GetOutputPort(0))
        self.half = encap()
        self.half.port = _cleaner.GetOutputPort()
        self.half.data = _cleaner.GetOutput()
        self.half.mapper = vtkPolyDataMapper()
        self.half.mapper.SetInputConnection(self.half.port)
        self.half.mapper.SetLookupTable(self.lut)
        self.half.actor = vtkActor()
        self.half.actor.SetMapper(self.half.mapper)
        self.ren.AddActor(self.half.actor)

        self.fig = plt.figure()
        self.ax = plt.gca()
        self.bar = self.ax.bar(range(histo_num_bins), np.zeros((histo_num_bins,)), width=1)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.clicked_on_bar)
        cid = self.fig.canvas.mpl_connect('button_release_event', self.released_on_bar)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)
        plt.ion()
        plt.show()

        self.whole.mapper.Update()
        self.half.mapper.Update()
        self.update_distance()
        self.ren_win.Render()


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
        origin = np.array(self.plane.GetOrigin())
        normal = np.array(self.plane.GetNormal())
        T = np.eye(4) - 2 * np.array([[*normal,0]]).T @ np.array([[ *normal, -origin.dot(normal) ]])
        self.mirror.GetInput().DeepCopy(np.array(T).ravel())

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

        arr, edgs = np.histogram(vtk_to_numpy(err), bins=histo_num_bins)
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
        from pycpd import DeformableRegistration        
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
        
        transform = vtkTransformPolyDataFilter()
        transform.SetTransform(self.mirror)
        transform.SetInputData(self.half.data)
        transform.Update()

        srcm = self.half.data
        src = transform.GetOutput()
        tar = self.whole.data
        reg_vtk = vtkPolyData()
        reg_vtk.DeepCopy(srcm)

        srcm =  polydata_to_numpy(srcm)
        src =  polydata_to_numpy(src)
        tar =  polydata_to_numpy(tar)
        
        reg = self.nicp(tar, srcm)

        # visualize this reg_vtk to see the result of nicp
        reg_vtk.GetPoints().SetData(numpy_to_vtk(reg.nodes))

        midpts = vtkPoints()
        midpts.SetData(numpy_to_vtk( src.nodes/2 + reg.nodes/2 ))
        origin, normal = [0.]*3, [0.]*3
        vtkPlane.ComputeBestFittingPlane(midpts, origin, normal)
        self.set_plane(origin, normal)


def main(stl_file_input):
    reader = vtkSTLReader()
    reader.SetFileName(stl_file_input)
    reader.Update()
    d = ChestVisualizer(reader.GetOutput())
    d.iren.Start()


    


if __name__ == '__main__':
    main(r'out.stl')
