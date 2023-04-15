## python packages
import sys, os, glob, json, math
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



from vtkmodules.vtkCommonColor import vtkNamedColors
colors = vtkNamedColors()
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkSmoothPolyDataFilter
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkDiscreteFlyingEdges3D, vtkTransformFilter
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkCommonCore import vtkPoints

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkDataSetAttributes
)
from vtkmodules.vtkFiltersCore import (
    vtkMaskFields,
    vtkThreshold,
    vtkWindowedSincPolyDataFilter
)
from vtkmodules.vtkFiltersGeneral import (
    vtkDiscreteFlyingEdges3D,
    vtkDiscreteMarchingCubes
)
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from vtkmodules.vtkImagingStatistics import vtkImageAccumulate
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkImagingMorphological import vtkImageOpenClose3D


# PySide
from PySide6.QtGui import QWindow, QKeyEvent
from PySide6.QtCore import Qt
from PySide6.QtCore import Qt, Signal, Slot, QEvent, QObject
from PySide6.Qt3DInput import Qt3DInput 
from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QMdiSubWindow, QMdiArea
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTK
from vtkmodules.vtkInteractionImage import vtkImageViewer2, vtkResliceImageViewer 
from enum import Enum, IntFlag, auto
import asyncio

AXIAL = vtkImageViewer2.SLICE_ORIENTATION_XY
CORONAL = vtkImageViewer2.SLICE_ORIENTATION_XZ
SAGITTAL = vtkImageViewer2.SLICE_ORIENTATION_YZ

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
        self.picked_point = [float('nan')]*3
        return None
    
    def pick(self, debug=True):
        pos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, self.GetDefaultRenderer())
        # record picked point position
        self.picked_point = [float('nan'),]*3
        self.picked_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            self.picked_point = self.picker.GetPickPosition()
            self.img_data.TransformPhysicalPointToContinuousIndex(self.picked_point, self.picked_ijk)
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = self.GetInteractor().objectName()
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.picked_point), end=' ')
        return None


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
                self.current_slice = tuple([int(round(x)) for x in self.picked_ijk])
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
    # each instance of this class is a vtk rendering window
    # this class is used to populate 2x2 central widget of the mainwindow
    # mouse event is first handled here by vtk, since
    # it binds closer to vtk, coordinates, objects, etc.
    # if it is only view related, handle it internally
    # then call parent for task-specific handling
    # for task such as setting slice for other subwindow
    # keyboard events are handled in qt -- passed to parents untouched
    # other event goes to base class implementation, vtk or qt
    # for mouse event handling, it is a design choice to
    # 1) extend this class
    # 2) create custom style
    # 3) simply attach callbacks

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



    def event(self, event):
        if isinstance(event, QKeyEvent):
            # handle key press together in 2x2 central window and not in subwindows
            self.parent().event(event)
            return True
        return super().event(event)


    

# class KeyFilter(QObject):
#     def eventFilter(self, obj, event):
#         if isinstance(event, QKeyEvent):
#             obj.parent().event(event)
#             return True
#         else:
#             # standard event processing
#             return super().eventFilter(obj, event)




class QVTK2x2Window(QWidget):
    # this class is where all vtk events are handled
    # mode is controled by its parent on view hierarchy thru delegation
    # 
    def __init__(self, *initargs):
        super().__init__(*initargs)
  
        # create four subviews
        # vtkImageViewer2.SLICE_ORIENTATION_YZ === 0
        # vtkImageViewer2.SLICE_ORIENTATION_XZ === 1
        # vtkImageViewer2.SLICE_ORIENTATION_XY === 2

        self.ctrl_pressed = False
        self.alt_pressed = False
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(.01)
        self.picked_point = [float('nan'),]*3
        self.picked_ijk = [float('nan'),]*3
        self.subviews = {}
        # three orthogonal views
        for orientation in (2,0,1):
            
            subview = QVTK(parent=self)
            subview.viewer = vtkImageViewer2()
            subview.viewer.SetRenderWindow(subview.window)
            subview.viewer.SetRenderer(subview.renderer)
            subview.viewer.SetSliceOrientation(orientation)
            subview.renderer.SetBackground(0.2, 0.2, 0.2)
            
            style = vtkInteractorStyleImage()
            style.AutoAdjustCameraClippingRangeOn()
            style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press_image)
            style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release_image)
            style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event_image)
            # style.mwf = style.AddObserver('MouseWheelForwardEvent', self.scroll_forward_image)
            # style.mwb = style.AddObserver('MouseWheelBackwardEvent', self.scroll_backward_image)

            style.SetDefaultRenderer(subview.renderer)
            subview.SetInteractorStyle(style)
            self.subviews[orientation] = subview
            
    
        # 3d view
        subview = QVTK(parent=self)
        subview.renderer.SetBackground(0.6863, 0.9333, 0.9333)
        
        style = vtkInteractorStyleTrackballCamera()
        style.AutoAdjustCameraClippingRangeOn()
        style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press_object)
        style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release_object)
        style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event_object)
        
        style.SetDefaultRenderer(subview.renderer)
        subview.SetInteractorStyle(style)
        self.subviews[3] = subview
        

        # put 4 subviews on a 2x2 grid
        self.gridlayout = QGridLayout(parent=self)
        self.gridlayout.setContentsMargins(0,0,0,0)
        self.gridlayout.addWidget(self.axial, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.sagittal, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.coronal, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.perspective, 1, 1, 1, 1)

        return None


    def show(self):
        # self.perspective.display_dummy()
        self.perspective.Render()
        self.perspective.show()
        self.axial.show()
        self.sagittal.show()
        self.coronal.show()
        super().show()

    @property
    def sagittal(self):
        return self.subviews[0]
    
    @property
    def coronal(self):
        return self.subviews[1]
    
    @property
    def axial(self):
        return self.subviews[2]
    
    @property
    def perspective(self):
        return self.subviews[3]
    
    def get_slice(self, orientation):
        return self.subviews[orientation].viewer.GetSlice()

    @Slot(int, int)
    def set_slice(self, orientation=None, new_slice=None):
        if orientation is None:
            self.set_slice(0)
            self.set_slice(1)
            self.set_slice(2)
            return None
        else:
            viewer = self.subviews[orientation].viewer
            min, max = viewer.GetSliceMin(), viewer.GetSliceMax()
            if new_slice is None:
                new_slice = min//2 + max//2
            if math.isnan(new_slice):
                return None
            new_slice = int(round(new_slice))
            if new_slice>=min and new_slice<=max:
                viewer.SetSlice(new_slice)
            else:
                print('out of bounds')
        return None


    def load_image(self, data):
        self.axial.viewer.SetInputData(data)
        self.sagittal.viewer.SetInputData(data)
        self.coronal.viewer.SetInputData(data)
        self.set_slice()
        return None


    def pick(self, pos, ren, debug=True):
        self.picker.Pick(pos[0], pos[1], 0, ren)
        picked_point = [float('nan'),]*3
        picked_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            picked_point = self.picker.GetPickPosition()
            self.parent().img_data.TransformPhysicalPointToContinuousIndex(picked_point, picked_ijk)
            self.picked_point, self.picked_ijk = picked_point, picked_ijk
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = 'change'
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.picked_point), end=' ')
        return None


    async def async_pick(self, pos, ren, debug=True):
        self.picker.Pick(pos[0], pos[1], 0, ren)
        picked_point = [float('nan'),]*3
        picked_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            picked_point = self.picker.GetPickPosition()
            self.parent().img_data.TransformPhysicalPointToContinuousIndex(picked_point, picked_ijk)
            self.picked_point, self.picked_ijk = picked_point, picked_ijk
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = 'change'
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.picked_point), end=' ')
        return None


    def left_button_press_image(self, obj, event):
        self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
        if not self.ctrl_pressed:
            obj.OnLeftButtonDown()

        return None


    def mouse_move_event_image(self, obj, event):
        self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
        if self.ctrl_pressed:
            self.set_slice(0, self.picked_ijk[0])
            self.set_slice(1, self.picked_ijk[1])
            self.set_slice(2, self.picked_ijk[2])
        else:
            obj.OnMouseMove()
        return None


    def left_button_release_image(self, obj, event):
        # do not rely on clean up
        # handle stuff on the fly using move event and implement abort
        obj.OnLeftButtonUp()

        return None
    
    def left_button_press_object(self, obj, event):
        # self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
        if not self.ctrl_pressed:
            obj.OnLeftButtonDown()

        return None


    def mouse_move_event_object(self, obj, event):
        asy = False
        if asy:
            if self.ctrl_pressed:
                if hasattr(self, 'move_loop'):
                    try:
                        self.move_loop.stop()
                    except:
                        pass
                self.move_loop = asyncio.new_event_loop()
                pos, ren = obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer()
                self.move_loop.run_until_complete(pick_point(self, pos, ren))
            else:
                obj.OnMouseMove()
            return None
        else:
            if self.ctrl_pressed:
                self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
                self.set_slice(0, self.picked_ijk[0])
                self.set_slice(1, self.picked_ijk[1])
                self.set_slice(2, self.picked_ijk[2])
            else:
                obj.OnMouseMove()
            return None


    def left_button_release_object(self, obj, event):
        # do not rely on clean up
        # handle stuff on the fly using move event and implement abort
        obj.OnLeftButtonUp()

        return None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if hasattr(event, 'key'):
            print(event.key())
            if event.key() == Qt.Key_Space.value:
                print("space pressed")
                return True
            if event.key() == Qt.Key_Control.value:
                print("ctrl pressed")
                self.ctrl_pressed = True
                return True
            if event.key() == Qt.Key_Alt.value:
                print("alt pressed")
                self.alt_pressed = True
                return True
        return False


    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if hasattr(event, 'key'):
            print(event.key())
            if event.key() == Qt.Key_Space.value:
                print("space released")
                return True
            if event.key() == Qt.Key_Control.value:
                print("ctrl released")
                self.ctrl_pressed = False
                return True
            if event.key() == Qt.Key_Alt.value:
                print("alt released")
                self.alt_pressed = False
                return True
        return False
        

class AppWindow(QMainWindow):

    def __init__(self, *args, **kw):

        # init super
        super().__init__(*args, **kw)
        
        # init parameters
        self.img_data = None
        self.lmk_data = None

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
        self.img_data = src.GetOutput()
        label = threshold_image(self.img_data,(0,1250))
        polyd = label_to_object(label)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polyd)
        actor = vtkActor()
        actor.SetMapper(mapper)
        self.central.perspective.renderer.AddActor(actor)
        self.central.load_image(self.img_data)
        # change the following line to use observer
        print('data loaded')
        self.setWindowTitle('VIEWING: ' + file)
        self.central.show()
        return None


    def create_polydata(thresh, )



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
    w.load_nifti(r'c:\py-scripts\n0001.nii.gz')
    sys.exit(app.exec())


async def pick_point(self, pos, ren):
    self.pick(pos, ren)            
    self.set_slice(0, self.picked_ijk[0])
    self.set_slice(1, self.picked_ijk[1])
    self.set_slice(2, self.picked_ijk[2])


def threshold_image(img_data, thresh):
    threshold = vtkImageThreshold()
    threshold.SetInputData(img_data)
    threshold.SetInValue(1.0)
    threshold.SetOutValue(0.0)
    threshold.ReplaceInOn()
    threshold.ReplaceOutOn()
    threshold.ThresholdBetween(*thresh)
    threshold.Update()
    return threshold.GetOutput()

def label_to_object(label_data):
    closer = vtkImageOpenClose3D()
    discrete_cubes = vtkDiscreteFlyingEdges3D()
    smoother = vtkWindowedSincPolyDataFilter()
    scalars_off = vtkMaskFields()
    geometry = vtkGeometryFilter()

    smoothing_iterations = 15
    pass_band = 0.001
    feature_angle = 120.0
    
    closer.SetInputData(label_data)
    closer.SetKernelSize(2, 2, 2)
    closer.SetCloseValue(1.0)

    discrete_cubes.SetInputConnection(closer.GetOutputPort())
    discrete_cubes.SetValue(0, 1.0)
    
    scalars_off.SetInputConnection(discrete_cubes.GetOutputPort())
    scalars_off.CopyAttributeOff(vtkMaskFields().POINT_DATA,
                                 vtkDataSetAttributes().SCALARS)
    scalars_off.CopyAttributeOff(vtkMaskFields().CELL_DATA,
                                 vtkDataSetAttributes().SCALARS)

    geometry.SetInputConnection(scalars_off.GetOutputPort())

    smoother.SetInputConnection(geometry.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    
    smoother.Update()
    return smoother.GetOutput()






if __name__ == '__main__':

    main(sys.argv)




