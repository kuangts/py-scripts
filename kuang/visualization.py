## python packages
import sys, os, glob, json
from abc import abstractmethod, ABC
## site packages
import numpy as np
from scipy.spatial  import KDTree
# vtk
import vtk
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
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
from PySide6.QtGui import QWindow
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QMdiSubWindow, QMdiArea
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTK

## my packages
# from .digitization import landmark


    # abstract classes

# class SubView(QMdiSubWindow):
    
#     # can be used as a regular qt window
#     def __init__(self, parent=None, interactor_style_type=None, frameless=True, **initargs):
        
#         super().__init__(parent=parent)
#         self.window = vtkSubWindow(interactor_style_type=interactor_style_type)
        
#         # install this window
#         wdgt = QWidget(self)
#         self.qvtk_interactor = QVTKRenderWindowInteractor(parent=wdgt, rw=self.window)
#         self.gridlayout = QGridLayout(wdgt)
#         self.gridlayout.setContentsMargins(0,0,0,0)
#         self.gridlayout.addWidget(self.qvtk_interactor, 0, 0, 1, 1)
#         self.window.renderer = vtkRenderer()
#         self.window.AddRenderer(self.window.renderer)
#         self.setWidget(wdgt)

#         # additional setup
#         if frameless:
#             self.setWindowFlag(Qt.FramelessWindowHint)        

#         return None

#     def show(self):
#         self.window.show()
#         super().show()

#     def quit(self):
#         pass


class QVTK(QVTK):
    
    # can be used as a regular qt window
    def __init__(self, parent=None, interactor_style_type=None, **initargs):
        self.window = vtkRenderWindow()
        initargs['rw'] = self.window
        super().__init__(parent=parent, **initargs)

        # set up interactor and style
        if interactor_style_type is None:
            interactor_style_type = vtkInteractorStyleTrackballCamera
        iren = vtkRenderWindowInteractor()
        style = interactor_style_type()
        style.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        style.picker = vtkCellPicker()
        style.picker.SetTolerance(0.05)
        iren.SetInteractorStyle(style)
        iren.SetRenderWindow(self.window)

        self.renderer = vtkRenderer()
        self.window.AddRenderer(self.renderer)
        style.SetDefaultRenderer(self.renderer)

        self.interactor = iren
        self.interactor_style = style
        return None

    def show(self):
        self.window.Render()
        self.interactor.Initialize()
        self.interactor.Start()
        return None

    def quit(self):
        self.window.Finalize()
        self.interactor.TerminateApp()
        del self
        return None
    

    # @property
    # def interactor(self): 
    #     return self.window.GetInteractor()
    
    # @property
    # def interactor_style(self): 
    #     return self.window.GetInteractor().GetInteractorStyle()

    def left_button_press_event(self, obj, event):
        pos = obj.GetInteractor().GetEventPosition()
        print(self.style == obj)
        print('left button pressed', pos)
        obj.picker.Pick(pos[0], pos[1], 0, self.ren)
        if obj.picker.GetCellId() != -1:
            coord = list(obj.picker.GetPickPosition())
            print(coord)

        return None


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




class OrthoView(QVTK):

    def __init__(self, parent=None, orientation=None, interactor_style_type=vtkInteractorStyleImage, **initargs):

        super().__init__(parent=parent, interactor_style_type=vtkInteractorStyleImage, **initargs)

        self.viewer = vtk.vtkImageViewer2()
        self.viewer.SetRenderWindow(self.window)
        self.viewer.SetRenderer(self.renderer)
        self.viewer.SetupInteractor(self.interactor)
        self.renderer.SetBackground(0.2, 0.2, 0.2)

        if orientation == 'axial':
            self.viewer.SetSliceOrientationToXY()
        elif orientation == 'sagittal':
            self.viewer.SetSliceOrientationToYZ()
        elif orientation == 'coronal':
            self.viewer.SetSliceOrientationToXZ()
        else:
            pass
        
        

class QVTK2x2Window(QWidget):

    def __init__(self, *initargs):

        super().__init__()
        
        self.axial = OrthoView(parent=self, orientation='axial')
        self.sagittal = OrthoView(parent=self, orientation='sagittal')
        self.coronal = OrthoView(parent=self, orientation='coronal')
        self.perspective = QVTK(parent=self)

        self.gridlayout = QGridLayout(parent=self)
        self.gridlayout.setContentsMargins(0,0,0,0)
        self.gridlayout.addWidget(self.axial, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.sagittal, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.coronal, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.perspective, 1, 1, 1, 1)

        return None


    @property
    def data_port(self):
        if hasattr(self, '_data_port'):
            return self._data_port
        else:
            return self.parent._data_port

    @data_port.setter
    def data_port(self, dp):
        self.axial.viewer.SetInputConnection(dp)
        self.sagittal.viewer.SetInputConnection(dp)
        self.coronal.viewer.SetInputConnection(dp)

    def show(self):

        self.axial.viewer.Render()
        self.axial.show()

        self.sagittal.viewer.Render()
        self.sagittal.show()

        self.coronal.viewer.Render()
        self.coronal.show()

        self.perspective.window.renderer.SetBackground(0.2, 0.2, 0.2)
        self.perspective.window.display_dummy()
        self.perspective.show()

        super().show()


class MyApplication(QApplication):
    windows = []
    def new_window(self):
        w = AppWindow()
        self.windows.append(w)
        w.show()
        return w


class AppWindow(QMainWindow):

    # def __init__(self, *args, **kw):
    #     super().__init__(*args, **kw)
    #     self.resize(640,480)
    #     self.setWindowTitle('NOTHING IS LOADED')
    #     self.central = Q2x2Window(self)
    #     # self.central.show()
    #     wdgt = QWidget(self)
    #     self.gridlayout = QGridLayout(wdgt)
    #     self.gridlayout.setContentsMargins(0,0,0,0)
    #     self.gridlayout.addWidget(self.central, 0, 0, 1, 1)
    #     self.setCentralWidget(self.central)
    #     for v in self.central.subWindowList():
    #         if isinstance(v, SubView) and not isinstance(v, OrthoSubView):
    #             v.renderer.SetBackground(0.6863, 0.9333, 0.9333)
    #     return None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.resize(640,480)
        self.setWindowTitle('NOTHING IS LOADED')
        self.central = QVTK2x2Window()
        self.setCentralWidget(self.central)


        # for v in self.central.subWindowList():
        #     if isinstance(v, SubView) and not isinstance(v, OrthoSubView):
        #         v.renderer.SetBackground(0.6863, 0.9333, 0.9333)
        return None


    def load_data(self, **kw):
        # bridge to ui
        if 'nifti_file' in kw:
            self.load_nifti(kw['nifti_file'])
        return None
    
    def load_nifti(self, file):
        src = vtkNIFTIImageReader()
        src.SetFileName(file)
        src.Update()
        self.data_port = src.GetOutputPort()
        # change the following line to use observer
        self.central.data_port = self.data_port
        print('data port connected')
        self.setWindowTitle('VIEWERING: ' + file)
        self.central.show()
        return None


def main(argv):
    app = MyApplication(argv)

    w = app.new_window()
    w.load_nifti(r'C:\data\pre-post-paired-40-send-1122\n0001\20110425-pre.nii.gz')

    # w.centralWidget().GetRenderWindow().renderer_axial.SetViewport(0,0,1,1)
    
    sys.exit(app.exec())


if __name__ == '__main__':

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

