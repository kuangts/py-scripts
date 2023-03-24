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
from vtkmodules.vtkInteractionImage import vtkImageViewer2, vtkResliceImageViewer 
from enum import Enum

## my packages
# from .digitization import landmark

class Mode(Enum):
    VIEW=0
    LANDMARK=1


class Actors(object):
    def __init__(self, renderer, **kw):
        super().__init__()
        self.renderer = renderer
        self.add(**kw)
    
    def add(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)
            self.renderer.AddActor(v)

    def remove(self, *names):
        for k in names:
            self.renderer.RemoveActor(getattr(self, k))
            delattr(self, k) 


class AppWidePara:
    # re-implement these at proper level
    # related to window hierarchy and event handling chain, not class inheritence
    @property
    def mode(self): pass

    @mode.setter
    def mode(self, m): pass

    @property
    def data_port(self): pass

    @data_port.setter
    def data_port(self, dp): pass

    def handle_key(self, key): pass


class ParaPassThru(AppWidePara):
    # Pass these app wide parameter through for getting and setting
    @property
    def mode(self):
        return self.parent().mode

    @mode.setter
    def mode(self, m):
        self.parent().mode = m

    @property
    def data_port(self):
        return self.parent().data_port

    @data_port.setter
    def data_port(self, dp):
        self.parent().data_port = dp

    def handle_key(self, key):
        self.parent().handle_key(key)
        return None


class QVTK(QVTK):

    def __init__(self, parent, renderer=None, **kw):
        super().__init__(parent=parent, **kw)
        if not renderer:
            renderer = vtkRenderer()
        self.renderer = renderer
        self.window.AddRenderer(self.renderer)
        self.actors = Actors(self.renderer)

        return None

    @property
    def window(self):
        return self.GetRenderWindow()

    def show(self):
        self.Initialize()
        self.window.Render()
        self.Start()
        super().show()

    def quit(self):
        self.window.Finalize()
        self.TerminateApp()
        del self

    def display_dummy(self):
        source = vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        self.actors.add(dummy=actor)
        # self.actors.remove('dummy')

    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is pressed')
        self.handle_key(key)

    def left_button_press(self, obj, event):
        pos = obj.GetInteractor().GetEventPosition()
        print('left button pressed', pos)
        obj.picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
        if obj.picker.GetCellId() != -1:
            print('on object')
            obj.clicked_and_not_moved = True
            obj.coord = list(obj.picker.GetPickPosition())
            obj.OnLeftButtonDown()
        return None
    
    def mouse_move_event(self, obj, event):
        if hasattr(obj, 'clicked_and_not_moved'):
            obj.clicked_and_not_moved = False
        obj.OnMouseMove()
        return None

    def left_button_release(self, obj, event):
        if hasattr(obj, 'clicked_and_not_moved'):
            if obj.clicked_and_not_moved:
                if self.mode == Mode.LANDMARK:
                    print(f'point coordinates {obj.coord}')
            del obj.clicked_and_not_moved, obj.coord
        obj.OnLeftButtonUp()
        return None


    def set_style(self, style_class=vtkInteractorStyleTrackballCamera):
        style = style_class()
        style.AutoAdjustCameraClippingRangeOn()
        style.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)
        style.picker = vtkCellPicker()
        style.picker.SetTolerance(0.1)
        style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press)
        style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release)
        style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event)
        style.kp = style.AddObserver('KeyPressEvent', self.key_press_event)
        self.SetInteractorStyle(style)
        return None


class SubView(QVTK, ParaPassThru): 
    def __init__(self, parent, **kw):
        super().__init__(parent=parent, **kw)
        self.renderer.SetBackground(0.6863, 0.9333, 0.9333)


class OrthoView(QVTK, ParaPassThru):

    def __init__(self, parent=None, orientation=None, **kw):

        super().__init__(parent=parent, **kw)

        self.viewer = vtkImageViewer2()
        self.viewer.SetRenderWindow(self.window)
        self.viewer.SetRenderer(self.renderer)

        self.renderer.SetBackground(0.2, 0.2, 0.2)
        
        if orientation == 'axial':
            self.viewer.SetSliceOrientationToXY()
        elif orientation == 'sagittal':
            self.viewer.SetSliceOrientationToYZ()
        elif orientation == 'coronal':
            self.viewer.SetSliceOrientationToXZ()
        else:
            pass

        return None

    def set_style(self, style_class=vtkInteractorStyleImage):
        super().set_style(style_class=style_class)
        style = self.GetInteractorStyle()
        style.mwf = style.AddObserver('MouseWheelForwardEvent', self.scroll_forward)
        style.mwb = style.AddObserver('MouseWheelBackwardEvent', self.scroll_backward)
        return None
    

    def scroll_forward(self, obj, event):
        min, max = self.viewer.GetSliceMin(), self.viewer.GetSliceMax()
        new_slice = self.viewer.GetSlice() + 1
        if new_slice>=min and new_slice<=max:
            self.viewer.SetSlice(new_slice)
        return None


    def scroll_backward(self, obj, event):
        min, max = self.viewer.GetSliceMin(), self.viewer.GetSliceMax()
        new_slice = self.viewer.GetSlice() - 1
        if new_slice>=min and new_slice<=max:
            self.viewer.SetSlice(new_slice)
        return None
        

class QVTK2x2Window(QWidget, ParaPassThru):

    def __init__(self, *initargs):
        
        super().__init__()
        
        # create four subviews
        self.axial = OrthoView(parent=self, orientation='axial')
        self.sagittal = OrthoView(parent=self, orientation='sagittal')
        self.coronal = OrthoView(parent=self, orientation='coronal')
        self.perspective = SubView(parent=self)

        # set up interaction for four subviews
        self.axial.set_style()
        self.sagittal.set_style()
        self.coronal.set_style()
        self.perspective.set_style()

        # put four 2x2 subviews on a grid
        self.gridlayout = QGridLayout(parent=self)
        self.gridlayout.setContentsMargins(0,0,0,0)
        self.gridlayout.addWidget(self.axial, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.sagittal, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.coronal, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.perspective, 1, 1, 1, 1)

        return None


    def show(self):

        self.axial.viewer.SetInputConnection(self.data_port)
        self.sagittal.viewer.SetInputConnection(self.data_port)
        self.coronal.viewer.SetInputConnection(self.data_port)

        self.axial.viewer.Render()
        self.axial.show()

        self.sagittal.viewer.Render()
        self.sagittal.show()

        self.coronal.viewer.Render()
        self.coronal.show()

        self.perspective.renderer.SetBackground(0.2, 0.2, 0.2)
        self.perspective.display_dummy()
        self.perspective.show()

        super().show()


class AppWindow(QMainWindow, AppWidePara):

    # APP PARAMETERS
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        self._mode = m

    @property
    def data_port(self):
        return self._data_port

    @data_port.setter
    def data_port(self, dp):
        self._data_port = dp

    def handle_key(self, key):
        if key.isdigit():
            self.mode = Mode(int(key))
        return None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.mode = Mode.VIEW
        self.data_port = None

        self.resize(640,480)
        self.setWindowTitle('NOTHING IS LOADED')
        self.central = QVTK2x2Window()
        self.setCentralWidget(self.central)

        return None

    # INTERFACE RELATED
    def load_data(self, **kw):
        # bridge to ui
        if 'nifti_file' in kw:
            self.load_nifti(kw['nifti_file'])
        return None
    
    # DATA MODEL RELATED
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
    

class MyApplication(QApplication):
    windows = []
    def new_window(self):
        w = AppWindow()
        self.windows.append(w)
        w.show()
        return w


def main(argv):
    app = MyApplication(argv)

    w = app.new_window()
    w.load_nifti(r'C:\data\pre-post-paired-40-send-1122\n0001\20110425-pre.nii.gz')
    
    sys.exit(app.exec())


if __name__ == '__main__':

    main(sys.argv)




