## python packages
import sys, os, glob, json

## site packages
import numpy as np
from scipy.spatial  import KDTree
# vtk
import vtk
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList
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
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
# PySide
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QMdiSubWindow, QMdiArea
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

## my packages
# from .digitization import landmark


class VtkCompatible:
    # renderer, interactor, actors (dictionary)
    def display_dummy(self):

        # Create source
        source = vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)

        # Create a mapper
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        # Create an actor
        actor = vtkActor()
        actor.SetMapper(mapper)

        self.add_actor(dummy=actor)

    def add_actor(self, **kwargs):
        if not hasattr(self, 'actors'):
            self.actors = {}
        for k,v, in kwargs.items():
            self.renderer.AddActor(v)
            self.actors[k] = v

    def remove_actor(self, *actor_names):
        for k in actor_names:
            self.renderer.RemoveActor(self.actors[k])
            del self.actors[k]


class Window(vtkRenderWindow, VtkCompatible):

    def __init__(self, interactor_style_type=None, size=None, title=None, **initargs):

        super().__init__()

        if interactor_style_type is None:
            interactor_style_type = vtkInteractorStyleTrackballCamera

        # set up renderer
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.6863, 0.9333, 0.9333)
        self.AddRenderer(self.renderer)

        # set up interactor and style
        self.interactor = vtkRenderWindowInteractor()
        self.interactor_style = interactor_style_type()
        self.interactor_style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(self.interactor_style)
        self.interactor.SetRenderWindow(self)

        # additional setup
        if size is not None:
            self.SetSize(*size)
        if title is not None:
            self.SetWindowName(title)
        return None

    def show(self):
        self.Render()
        self.interactor.Initialize()
        self.interactor.Start()

    def quit(self):
        self.Finalize()
        self.interactor.TerminateApp()
        del self


class vtkSubView(QMdiSubWindow, VtkCompatible):
    # can be used as a regular qt window
    def __init__(self, parent=None, interactor_style_type=None, size=None, title=None, frameless=True, **initargs):

        super().__init__(parent)
        
        wdgt = QWidget(self)
        self.vtkWidget = QVTKRenderWindowInteractor(wdgt)
        self.gridlayout = QGridLayout(wdgt)
        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 1, 1)
        self.setWidget(wdgt)

        # set up renderer
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.6863, 0.9333, 0.9333)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # set up interactor
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        if interactor_style_type is None:
            interactor_style_type = vtkInteractorStyleTrackballCamera
        self.interactor_style = interactor_style_type()
        self.interactor_style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(self.interactor_style)

        # additional setup
        if frameless:
            self.setWindowFlag(Qt.FramelessWindowHint)        
        if size is not None:
            self.resize(*size)
        if title is not None:
            self.setObjectName(title)
        return None


    def show(self):
        super().show()
        self.interactor.Initialize()


    def quit(self):
        pass



class Q2x2Window(QMdiArea):
    def __init__(self, size=None, title=None, **initargs):

        super().__init__()

        for _ in range(4):
            sub = vtkSubView()
            sub.display_dummy()
            sub.show()
            self.addSubWindow(sub)

        self.tileSubWindows()

        if size is not None:
            self.resize(*size)
        if title is not None:
            self.setObjectName(title)
        return None

    def show(self):
        super().show()

    @property
    def axial(self):
        self.subWindowList()[0]

    @property
    def sagittal(self):
        self.subWindowList()[1]

    @property
    def coronal(self):
        self.subWindowList()[2]

    @property
    def perspective(self):
        self.subWindowList()[3]


def main(argv):
    app = QApplication(argv)
    w = QMainWindow()
    mdi = Q2x2Window()
    w.setCentralWidget(mdi)
    w.show()
    sys.exit(app.exec())



if __name__ == "__main__":

    main(sys.argv)

















# lmk_lib = landmark.Library()
# colors = vtkNamedColors()


# class Model:
#     def __init__(self, data_or_dataport, make_actor=True, color=colors.GetColor3d('yellow'), transform=None):
#         # transform Model should only affect said Model, unless otherwise intended
#         # set_matrix on MatrixTransform should affect all models using the same instance

#         if transform is None:
#             transform = MatrixTransform()
#         T = vtkTransformFilter()
#         T.SetTransform(transform)
#         if isinstance(data_or_dataport, vtkAlgorithmOutput):
#             T.SetInputConnection(data_or_dataport)
#             T.Update()
#             self.inputport = data_or_dataport
#         elif isinstance(data_or_dataport, vtkPolyData):
#             T.SetInputData(data_or_dataport)
#             T.Update()
#             self.inputport = None
#         else:
#             raise ValueError('Model: wrong type of data')
#         self.outputport = T.GetOutputPort()
#         # self.outputport.SetProducer(T)
#         self.filter = T

#         if make_actor:
#             mapper = vtkPolyDataMapper()
#             mapper.SetInputConnection(T.GetOutputPort())
#             mapper.Update()
#             actor = vtkActor()
#             actor.SetMapper(mapper)
#             self.actor = actor
#             self.actor.GetProperty().SetColor(*color)
#         else:
#             self.actor = None


#     def set_transform(self, T):
#         if not isinstance(T, MatrixTransform):
#             raise ValueError('set_transform only accepts MatrixTransform instance')
#         self.filter.SetTransform(T)
#         self.filter.Update()

#     def obbtree(self):
#         pass

# class MatrixTransform(vtkMatrixToLinearTransform): # vtk uses pre-multiplication
#     # convenience class to make it work with both numpy and vtk
#     # and work for both update and reset
#     def __init__(self, T=np.eye(4)):
#         if isinstance(T, vtkMatrix4x4):
#             _T = T
#         else:
#             _T = vtkMatrix4x4()
#             _T.DeepCopy(np.array(T).ravel())

#         self.SetInput(_T)
#         self.Update()
    
#     @property
#     def matrix(self):
#         return self.GetInput()

#     def update_matrix(self, T):
#         if isinstance(T, vtkMatrix4x4):
#             self.matrix.DeepCopy(T)
#         else:
#             self.matrix.DeepCopy(np.array(T).ravel())

#         self.Update()

#     def set_matrix(self, T):
#         # each set call resets back storage matrix
#         # so other vtkTransform will not be affected
#         if isinstance(T, vtkMatrix4x4):
#             _T = T
#         else:
#             _T = vtkMatrix4x4()
#             _T.DeepCopy(np.array(T).ravel())
#             self.SetInput(_T)

#         self.Update()


# class Visualizer():

#     @staticmethod
#     def load_landmark(file, ordered_list=None):
#         # read existing file if any
#         if os.path.isfile(file):
#             if file.endswith('.cass'):
#                 lmk = LandmarkDict.from_cass(file)
#             else:
#                 lmk = LandmarkDict.from_text(file)
#         else:
#             lmk = LandmarkDict()

#         # keep only required landmarks
#         if ordered_list is not None:
#             lmk = lmk.select(ordered_list)

#         return lmk



#     @staticmethod
#     def generate_mdoel_from_volume(source, preset=None, threshold=(0,0), color=(0,0,0), transform=vtkMatrix4x4(), **passthrough):
#         if preset == 'bone':
#             threshold = (0,1250)
#             color=colors.GetColor3d('grey')
#         elif preset == 'skin':
#             threshold=(0, 324)
#             color=colors.GetColor3d('peachpuff')

#         extractor = vtkFlyingEdges3D()
#         extractor.SetInputConnection(source.GetOutputPort())
#         extractor.SetValue(*threshold)
#         extractor.Update()
#         nifti_origin_translation = vtkTransformFilter()
#         T = np.eye(4)

#         try: # if nifti
#             mats = source.GetSFormMatrix()
#             matq = source.GetQFormMatrix()
#             ts = np.eye(4)
#             tq = np.eye(4)
#             mats.DeepCopy(ts.ravel(),mats)
#             matq.DeepCopy(tq.ravel(),matq)
#             assert np.isclose(ts, tq).all(), "check NIFTI file, sform and qform matrices are different"
#             T[:3,3:] = ts[:3,:3] @ ts[:3,3:]
#         except AssertionError as e:
#             raise e
            
#         nifti_origin_translation.SetTransform(MatrixTransform(T))
#         nifti_origin_translation.SetInputConnection(extractor.GetOutputPort())
#         nifti_origin_translation.Update()

#         return Model(nifti_origin_translation.GetOutputPort(), color=color)

#     @staticmethod
#     def trace(point, direction, target_obbtree, option='normal'):
#         # in this program, ray-tracing always happens from self.bone_actor to self.skin_actor
#         # options: normal, closest, hybrid
                
#         points = vtkPoints()
#         cellIds = vtkIdList()
#         coord = [float('nan')]*3

#         if option == 'normal':
#             code = target_obbtree.IntersectWithLine(point, [x[0]+x[1]*50 for x in zip(point,direction)], points, cellIds)
#             pointData = points.GetData()
#             intersections = [ pointData.GetTuple3(idx) for idx in range(pointData.GetNumberOfTuples()) ]
#             if len(intersections):
#                 vec = np.asarray(intersections)-np.asarray(point)
#                 signed_distance = vec.dot(direction)
#                 ind = np.argmax(signed_distance)
#                 if signed_distance[ind] > 0:
#                     coord = intersections[ind]

#         elif option == 'closest':
#             print('use closest_point instead')

#         point[0],point[1],point[2] = coord[0],coord[1],coord[2]


#     @staticmethod
#     def closest_point(point, direction, target_kdtree, guided_by_normal=True, normal_scale_factor=1.):     
#         if guided_by_normal:
#             d, _ = target_kdtree.query(point, k=1)
#             tr = KDTree(np.hstack((target_kdtree.data, target_kdtree.data-np.array([point]))))
#             d, id = tr.query([*point, *map(lambda x:x*d, direction)], k=1)
#         else:
#             tr = target_kdtree
#             d, id = tr.query(point, k=1)

#         coord = tr.data[id]
#         point[0],point[1],point[2] = coord[0],coord[1],coord[2]
        

#     def draw_landmark(self, lmk, renderer=None, **kwargs):
#         vtk_lmk = [ vtkLandmark(k, v, **kwargs) for k,v in lmk.items() ]
#         for l in vtk_lmk:
#             l.set_renderer(renderer if renderer is not None else self.ren)
#         return vtk_lmk


#     def __init__(self):

#         # create window 
#         renderer = vtkRenderer()
#         renderer.SetBackground(colors.GetColor3d('paleturquoise'))
#         ren_win = vtkRenderWindow()
#         ren_win.AddRenderer(renderer)
#         ren_win.SetSize(640, 480)
#         ren_win.SetWindowName('Lip Landmarks')
#         iren = vtkRenderWindowInteractor()
#         iren.SetRenderWindow(ren_win)
#         style = vtkInteractorStyleTrackballCamera()
#         style.SetDefaultRenderer(renderer)
#         iren.SetInteractorStyle(style)

#         self.ren = renderer
#         self.ren_win = ren_win
#         self.iren = iren
#         self.style = style
#         self.actors = {}

#         # text box for status update, soon to be replaced by PyQt
#         self.status = vtkTextActor()
#         # self.status.SetPosition2(10, 40)
#         self.status.GetTextProperty().SetFontSize(16)
#         self.status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
#         self.ren.AddActor2D(self.status)

#         self.ren_win.AddObserver('ModifiedEvent', self.position_panel)
#         self.position_panel()

#     def start(self):
#         self.iren.Initialize()
#         self.ren_win.Render()
#         self.iren.Start()

#     def position_panel(self, obj=None, event=None):
#         s = self.ren_win.GetSize()
#         s0 = [float('nan')]*2
#         self.status.GetSize(self.ren, s0)
#         # self.status.SetPosition(s[0]*.8,s[1]*.9-s0[1])
#         self.status.SetPosition(0,0)

#     def add_actor(self, renderer=None, **actor_dict):
#         if renderer is None:
#             renderer = self.ren
#         for k,v in actor_dict.items():
#             renderer.AddActor(v)
#             self.actors[k] = v

#     def remove_actor(self, renderer=None, *actor_names):
#         if renderer is None:
#             renderer = self.ren
#         for k in actor_names:
#             renderer.RemoveActor(self.actors[k])
#             del self.actors[k]

