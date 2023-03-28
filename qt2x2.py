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
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QMdiSubWindow, QMdiArea
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTK
from vtkmodules.vtkInteractionImage import vtkImageViewer2, vtkResliceImageViewer 
from enum import Enum, IntFlag, auto
import asyncio

## my packages
# from .digitization import landmark

class Mode(Enum):
    VIEW=0
    LANDMARK=1

class WindowParameter: 
    parameter_keys = {
        'mode',
        'interaction_state',
        'img_data',
        'lmk_data'
    }

class InteractionState(IntFlag):
    # bits assignment
    REST = 0
    MOVED = auto()
    LEFT_BUTTON = auto()
    RIGHT_BUTTON = auto()
    CTRL = auto()
    ALT = auto()

    # derived
    LEFT_CLICK = LEFT_BUTTON & ~MOVED
    RIGHT_CLICK = RIGHT_BUTTON & ~MOVED

    def mouse_moved(self):
        print(f'mouse moved', end=' ')
        return self.__class__(self.value | self.__class__.MOVED)

    def left_button_pressed(self):
        # set left-button bit and clear moved bit
        print(f'left button pressed', end=' ')
        return self.__class__((self.value | self.LEFT_BUTTON) & ~self.MOVED)

    def left_button_released(self):
        # clear left-button bit and moved bit
        print(f'left button released', end=' ')
        return self.__class__(self.value & ~self.LEFT_BUTTON & ~self.MOVED)

    def right_button_pressed(self):
        # set right-button bit and clear moved bit
        print(f'right button pressed', end=' ')
        return self.__class__((self.value | self.RIGHT_BUTTON) & ~self.MOVED)

    def right_button_released(self):
        # clear right-button bit and moved bit
        print(f'right button released', end=' ')
        return self.__class__(self.value & ~self.RIGHT_BUTTON & ~self.MOVED)
  

class EventHandler:

    def key_press_event(self, obj, event): 
        obj.OnKeyPress()
        return None
    
    def key_release_event(self, obj, event):
        obj.OnKeyRelease()
        return None
    
    # button press is relatively expensive
    def left_button_press(self, obj, event):
        self.interaction_state = self.interaction_state.left_button_pressed()
        obj.OnLeftButtonDown()
        return None
    
    # mouse move could abort
    def mouse_move_event(self, obj, event):
        self.interaction_state = self.interaction_state.mouse_moved()
        obj.OnMouseMove()
        return None
    
    # do not rely on clean up after release
    # handle stuff on the fly using move event and implement abort
    def left_button_release(self, obj, event):
        self.interaction_state = self.interaction_state.left_button_released()
        obj.OnLeftButtonUp()
        return None



class Actors(object):
    def __init__(self, renderer, **kw):
        super().__init__()
        self.renderer = renderer
        self.add(**kw)
    
    def add(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)
            v.SetObjectName(k)
            self.renderer.AddActor(v)

    def remove(self, *names):
        for k in names:
            self.renderer.RemoveActor(getattr(self, k))
            delattr(self, k) 




    # @property
    # def mode(self):
    #     return self._mode

    # @mode.setter
    # def mode(self, m):
    #     self._mode = m

    # @property
    # def interaction_state(self):
    #     return self.root._interaction_state

    # @interaction_state.setter
    # def interaction_state(self, i):
    #     self.root._interaction_state = i

    # @property
    # def current_slice(self):
    #     return self.root._current_slice
    
    # @current_slice.setter
    # def current_slice(self, ijk_tuple): # use tuple to prevent change to member
    #     self.root._current_slice = ijk_tuple

    # @property
    # def img_data(self):
    #     return self.root._img_data

    # @img_data.setter
    # def img_data(self, data):
    #     self.root._img_data = data
        

class MyStyleBase(vtkInteractorStyleTrackballCamera):

    def __init__(self) -> None:
        super().__init__()
        self.AutoAdjustCameraClippingRangeOn()
        self.lbp = self.AddObserver('LeftButtonPressEvent', self.left_button_press)
        self.lbr = self.AddObserver('LeftButtonReleaseEvent', self.left_button_release)
        self.lbr = self.AddObserver('MouseMoveEvent', self.mouse_move_event)
        self.kp = self.AddObserver('KeyPressEvent', self.key_press_event)
        self.kr = self.AddObserver('KeyReleaseEvent', self.key_release_event)
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.1)
        self.current_point = [float('nan')]*3
        return None
    
    def pick(self, debug=True):
        pos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, self.GetDefaultRenderer())
        # record picked point position
        self.current_point = [float('nan'),]*3
        self.current_point_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            self.current_point = self.picker.GetPickPosition()
            self.img_data.TransformPhysicalPointToContinuousIndex(self.current_point, self.current_point_ijk)
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = self.GetInteractor().objectName()
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.current_point), end=' ')
        return None


    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is pressed')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state | InteractionState.CTRL
        elif key.startswith('Alt'):
            self.interaction_state = self.interaction_state | InteractionState.ALT
        else:
            self.parent().key_press_event(obj, event)


    def key_release_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is released')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state & ~InteractionState.CTRL
        if key.startswith('Alt'):
            self.interaction_state = self.interaction_state & ~InteractionState.ALT
        else:
            self.parent().key_release_event(obj, event)


    # button press/release is relatively expensive
    # mouse move could abort
    def left_button_press(self, obj, event):
        self.interaction_state = self.interaction_state.left_button_pressed()
        self.pick(obj)
        if self.app_window.mode == Mode.LANDMARK:
            print('& LMK', end=' ')
        elif self.app_window.mode == Mode.VIEW:
            if self.interaction_state & InteractionState.ALT & ~InteractionState.CTRL:
                obj.OnLeftButtonDown()
    
        print('**')
        return None


    def mouse_move_event(self, obj, event):
        self.interaction_state = self.interaction_state.mouse_moved()
        # if self.current_ui_task is not None:
        #     self.current_ui_task.cancel()
        # self.current_ui_task = asyncio.create_task(self.pick(obj))
        # await self.current_ui_task
        self.pick(obj)
        if self.interaction_state & InteractionState.CTRL:
            try:
                self.current_slice = tuple([int(round(x)) for x in self.current_point_ijk])
            except: pass # some nan situation
            print('**')
            return None

        elif self.interaction_state & InteractionState.LEFT_BUTTON:
            if self.app_window.mode == Mode.LANDMARK:
                print('& LMK', end=' ')
            elif self.app_window.mode == Mode.VIEW:
                print('& VIEW', end=' ')
                obj.OnMouseMove() # might do nothing if OnLeftButtonDown is not properly called
                    
        print('**')
        return None


    def left_button_release(self, obj, event):
        # do not rely on clean up
        # handle stuff on the fly using move event and implement abort
        obj.OnLeftButtonUp()
        self.interaction_state = self.interaction_state.left_button_released()

        print('**')
        return None



class QVTK(QVTK):

    def __init__(self, parent=None, renderer=None, **kw):
        super().__init__(parent=parent, **kw)
        if not renderer:
            renderer = vtkRenderer()
        self.renderer = renderer
        self.window.AddRenderer(self.renderer)
        self.actors = Actors(self.renderer)

        return None

    @property
    def app_window(self):
        app_window = self
        if hasattr(app_window, 'parent'):
            app_window = self.parent()
        else:
            return app_window

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



class ObjectView(QVTK, EventHandler):
    def __init__(self, parent=None, **kw):
        super().__init__(parent=parent, **kw)
        style = vtkInteractorStyleTrackballCamera()
        style.AutoAdjustCameraClippingRangeOn()
        style.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)
        style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press)
        style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release)
        style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event)
        style.kp = style.AddObserver('KeyPressEvent', self.key_press_event)
        style.kr = style.AddObserver('KeyReleaseEvent', self.key_release_event)

        self.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)
        return None


    @property
    def current_ui_task(self):
        if hasattr(self, '_current_ui_task'):
            return self._current_ui_task
        else:
            return None
    
    @current_ui_task.setter
    def current_ui_task(self, t):
        setattr(self, '_current_ui_task', t)
        return None


    def pick(self, obj, debug=True):
        pos = obj.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
        # record picked point position
        self.current_point = [float('nan'),]*3
        self.current_point_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            self.current_point = self.picker.GetPickPosition()
            self.img_data.TransformPhysicalPointToContinuousIndex(self.current_point, self.current_point_ijk)
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = self.objectName()
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.current_point), end=' ')
        return None


    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is pressed')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state | InteractionState.CTRL
        elif key.startswith('Alt'):
            self.interaction_state = self.interaction_state | InteractionState.ALT
        else:
            self.parent().key_press_event(obj, event)


    def key_release_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is released')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state & ~InteractionState.CTRL
        if key.startswith('Alt'):
            self.interaction_state = self.interaction_state & ~InteractionState.ALT
        else:
            self.parent().key_release_event(obj, event)


    # button press is relatively expensive
    # mouse move could abort
    def left_button_press(self, obj, event):
        self.interaction_state = self.interaction_state.left_button_pressed()
        self.pick(obj)
        if self.app_window.mode == Mode.LANDMARK:
            print('& LMK', end=' ')
        elif self.app_window.mode == Mode.VIEW:
            if self.interaction_state & InteractionState.ALT & ~InteractionState.CTRL:
                obj.OnLeftButtonDown()
    
        print('**')
        return None


    def mouse_move_event(self, obj, event):
        self.interaction_state = self.interaction_state.mouse_moved()
        # if self.current_ui_task is not None:
        #     self.current_ui_task.cancel()
        # self.current_ui_task = asyncio.create_task(self.pick(obj))
        # await self.current_ui_task
        self.pick(obj)
        if self.interaction_state & InteractionState.CTRL:
            print('**')
            return None

        elif self.interaction_state & InteractionState.LEFT_BUTTON:
            if self.app_window.mode == Mode.LANDMARK:
                print('& LMK', end=' ')
            elif self.app_window.mode == Mode.VIEW:
                print('& VIEW', end=' ')
                obj.OnMouseMove() # might do nothing if OnLeftButtonDown is not properly called
                    
        print('**')
        return None


    def left_button_release(self, obj, event):
        # do not rely on clean up
        # handle stuff on the fly using move event and implement abort
        obj.OnLeftButtonUp()
        self.interaction_state = self.interaction_state.left_button_released()

        print('**')
        return None

class ImageView(QVTK, EventHandler):

    slice_change_singal = Signal(int, int) # orientation, slice#
    all_slice_change_singal = Signal(int, int, int) # YZ slice, XZ slice, ZY slice

    def __init__(self, parent=None, orientation=2, **kw):
        # vtkImageViewer2.SLICE_ORIENTATION_YZ === 0
        # vtkImageViewer2.SLICE_ORIENTATION_XZ === 1
        # vtkImageViewer2.SLICE_ORIENTATION_XY === 2
        super().__init__(parent=parent, **kw)

        self.orientation = orientation

        style = vtkInteractorStyleImage()
        style.AutoAdjustCameraClippingRangeOn()
        style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press)
        style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release)
        style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event)
        style.kp = style.AddObserver('KeyPressEvent', self.key_press_event)
        style.kr = style.AddObserver('KeyReleaseEvent', self.key_release_event)
        style.mwf = style.AddObserver('MouseWheelForwardEvent', self.scroll_forward)
        style.mwb = style.AddObserver('MouseWheelBackwardEvent', self.scroll_backward)

        style.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)
        self.viewer = vtkImageViewer2()
        self.viewer.SetRenderWindow(self.window)
        self.viewer.SetRenderer(self.renderer)
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        
        if orientation == vtkImageViewer2.SLICE_ORIENTATION_YZ:
            self.viewer.SetSliceOrientationToYZ()
            self.setObjectName('sagittal')
        elif orientation == vtkImageViewer2.SLICE_ORIENTATION_XZ:
            self.viewer.SetSliceOrientationToXZ()
            self.setObjectName('coronal')
        elif orientation == vtkImageViewer2.SLICE_ORIENTATION_XY:
            self.viewer.SetSliceOrientationToXY()
            self.setObjectName('axial')
        else:
            raise ValueError('wrong orientation')

        return None

    
    def scroll_forward(self, obj, event):
        self.slice_change_singal.emit(self.orientation, self.viewer.GetSlice() + 1)
        return None


    def scroll_backward(self, obj, event):
        self.slice_change_singal.emit(self.orientation, self.viewer.GetSlice() - 1)
        return None
        

    def show(self):
        self.viewer.SetInputData(self.img_data)
        self.viewer.SetSlice((self.viewer.GetSliceMin() + self.viewer.GetSliceMax())//2)
        self.viewer.Render()
        super().show()

    def pick(self, obj, debug=True):
        pos = obj.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
        # record picked point position
        self.current_point = [float('nan'),]*3
        self.current_point_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            self.current_point = self.picker.GetPickPosition()
            self.img_data.TransformPhysicalPointToContinuousIndex(self.current_point, self.current_point_ijk)
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = self.objectName()
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.current_point), end=' ')
        return None


    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is pressed')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state | InteractionState.CTRL
        elif key.startswith('Alt'):
            self.interaction_state = self.interaction_state | InteractionState.ALT
        else:
            self.parent().key_press_event(obj, event)


    def key_release_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is released')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state & ~InteractionState.CTRL
        if key.startswith('Alt'):
            self.interaction_state = self.interaction_state & ~InteractionState.ALT
        else:
            self.parent().key_release_event(obj, event)



    def left_button_press(self, obj, event):
        self.app_window.interaction_state = self.interaction_state.left_button_pressed()
        self.pick(obj)
        if self.app_window.param.mode == Mode.LANDMARK:
            print('& LMK', end=' ')
        elif self.app_window.mode == Mode.VIEW:
            if self.interaction_state & InteractionState.ALT & ~InteractionState.CTRL:
                obj.OnLeftButtonDown()
    
        print('**')
        return None


    def mouse_move_event(self, obj, event):
        self.interaction_state = self.interaction_state.mouse_moved()
        # if self.current_ui_task is not None:
        #     self.current_ui_task.cancel()
        # self.current_ui_task = asyncio.create_task(self.pick(obj))
        # await self.current_ui_task
        self.pick(obj)
        if self.interaction_state & InteractionState.CTRL:
            ijk = [int(round(x)) for x in self.current_point_ijk]
            self.all_slice_change_singal.emit(self.)
            print('**')
            return None

        elif self.interaction_state & InteractionState.LEFT_BUTTON:
            if self.app_window.mode == Mode.LANDMARK:
                print('& LMK', end=' ')
            elif self.app_window.mode == Mode.VIEW:
                print('& VIEW', end=' ')
                obj.OnMouseMove() # might do nothing if OnLeftButtonDown is not properly called
                    
        print('**')
        return None


    def left_button_release(self, obj, event):
        # do not rely on clean up
        # handle stuff on the fly using move event and implement abort
        obj.OnLeftButtonUp()
        self.interaction_state = self.interaction_state.left_button_released()

        print('**')
        return None



class QVTK2x2Window(QWidget):

    def __init__(self, *initargs):
        super().__init__(*initargs)
        
        # create four subviews
        self.orthoviews = [
            ImageView(parent=self, orientation=vtkImageViewer2.SLICE_ORIENTATION_YZ),
            ImageView(parent=self, orientation=vtkImageViewer2.SLICE_ORIENTATION_XZ),
            ImageView(parent=self, orientation=vtkImageViewer2.SLICE_ORIENTATION_XY),
        ]

        self.interaction_param = InteractionState.REST

        # keep current with slice changes
        for v in self.orthoviews:
            v.slice_change_singal.connect(self.set_slice)

        self.perspective = ObjectView(parent=self)

        # put 4 subviews on a 2x2 grid
        self.gridlayout = QGridLayout(parent=self)
        self.gridlayout.setContentsMargins(0,0,0,0)
        self.gridlayout.addWidget(self.axial, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.sagittal, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.coronal, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.perspective, 1, 1, 1, 1)

        return None


    def show(self):
        self.perspective.renderer.SetBackground(0.6863, 0.9333, 0.9333)
        self.perspective.display_dummy()
        self.perspective.Render()
        self.perspective.show()
        for v in self.orthoviews: v.show()
        super().show()

    @property
    def sagittal(self):
        return self.orthoviews[0]
    
    @property
    def coronal(self):
        return self.orthoviews[1]
    
    @property
    def axial(self):
        return self.orthoviews[2]
    
    def get_slice(self, orientation):
        return self.central.orthoviews[orientation].viewer.GetSlice()

    @Slot(int, int)
    def set_slice(self, orientation, new_slice):
        viewer = self.central.orthoviews[orientation].viewer
        min, max = viewer.GetSliceMin(), viewer.GetSliceMax()
        if new_slice>=min and new_slice<=max:
            viewer.SetSlice(new_slice)
        return None

    def handle_key(self, key):
        if key.isdigit():
            self.mode = Mode(int(key))
        elif key.startswith('Control'):
            self.interaction_state = self.interaction_state | InteractionState.CTRL
        elif key.startswith('Alt'):
            self.interaction_state = self.interaction_state | InteractionState.ALT

        return None



class AppWindow(QMainWindow):

    def __init__(self, *args, **kw):

        # init super
        super().__init__(*args, **kw)
        
        # init parameters
        self.param = WindowParameter()
        self.param.img_data = None
        self.param.lmk_data = None
        self.param.mode = Mode.VIEW
        self.param.interaction_state = InteractionState.REST

        # init children - main 2x2 window + side bar
        self.setWindowTitle('NOT LOADED')
        self.central = QVTK2x2Window(self)
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
        self.image_data = src.GetOutput()
        # change the following line to use observer
        print('data source connected')
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




