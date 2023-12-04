## python packages
import sys, os, glob, json, math
from abc import abstractmethod, ABC
from collections.abc import Collection
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
from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout, QVBoxLayout, QWidget, QMdiSubWindow, QMdiArea, QDockWidget, QTreeWidgetItem, QTreeWidget, QLabel
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionImage import vtkImageViewer2, vtkResliceImageViewer, vtkResliceImageViewer 
from enum import Enum, IntFlag, auto
import numbers
import asyncio



AXIAL = vtkResliceImageViewer.SLICE_ORIENTATION_XY
CORONAL = vtkResliceImageViewer.SLICE_ORIENTATION_XZ
SAGITTAL = vtkResliceImageViewer.SLICE_ORIENTATION_YZ
LEFT = 0
RIGHT = 1



class DataManager:

    @property
    def root(self):
        root = self
        while hasattr(root, 'parent'):
            if root.parent() is None:
                return root
            else:
                root = root.parent()



    def load_nifti(self, file):
        src = vtkNIFTIImageReader()
        src.SetFileName(file)
        src.Update()
        self.image_data = src.GetOutput()
        # self.load_image(self.image_data)
        # label = threshold_image(self.image_data,(0,1250))
        # polyd = label_to_object(label)
        # mapper = vtkPolyDataMapper()
        # mapper.SetInputData(polyd)
        # actor = vtkActor()
        # actor.SetMapper(mapper)
        # self.central.perspective.renderer.AddActor(actor)
        # change the following line to use observer
        print('data loaded')
        self.setWindowTitle('VIEWING: ' + file)
        self.central.show()
        return None


    @property
    def image_data(self):
        root_obj = self.root
        if self == root_obj:
            if not hasattr(self, '_image_data'):
                setattr(self, '_image_data', None)
            return self._image_data
        else:
            return root_obj.image_data


    @image_data.setter
    def image_data(self, data):
        if self.image_data != data:
            # send image changed signal
            pass
        self.root._image_data = data
        return None


class EventHandler:
    '''for collective event handling, such as coordinated zoom, or reslice
    event for each individual views is handled from within the views, not here
    has one property - views, which contains all QVTK children under its control
    the methods will be invoked most likely by styles of the QVTK children
    '''

    @property
    def root(self):
        root = self
        while hasattr(root, 'parent'):
            if root.parent() is None:
                return root
            else:
                root = root.parent()



    def reslice(self, new_coordinates=(None,)*3):
        print(new_coordinates)
        # slice all views at the same time
        if not isinstance(self, DataManager) or not self.image_data or not hasattr(self, 'children'):
            return None
        
        new_slice = [float('nan')]*3
        old_slice = new_slice.copy()
        self.image_data.TransformPhysicalPointToContinuousIndex(new_coordinates, new_slice)

        for view in self.children():

            if hasattr(view, 'reslice'):
                view.reslice(new_slice)
                continue
            
            elif hasattr(view, 'viewer') and hasattr(view, 'id'):

                id = view.id
                s = new_slice[id]

                if s is None:
                    s = view.viewer.GetSliceMin()/2 + view.viewer.GetSliceMax()/2
                
                if not isinstance(s, numbers.Number) or s<view.viewer.GetSliceMin() and s>view.viewer.GetSliceMax():
                    old_slice[id] = None

                old_slice[id] = view.viewer.GetSlice()
                view.viewer.SetSlice(int(round(s)))
                
        return old_slice


class QVTK(QVTKRenderWindowInteractor):
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
    # 2) preferred: create custom style
    # 3) simply attach callbacks

    def __init__(self, parent=None, renderer=None, **kw):
        super().__init__(parent=parent, **kw)
        if not renderer:
            renderer = vtkRenderer()
        self.renderer = renderer
        self.window.AddRenderer(self.renderer)
        return None

    @property
    def window(self):
        return self.GetRenderWindow()

    @property
    def id(self):
        return ''

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



class SliceViewer(QVTK):
    
    def __init__(self, parent=None, renderer=None, cursor=None, **kw):

        super().__init__(parent=parent, renderer=renderer, **kw)
        
        self.viewer = vtkResliceImageViewer()
        self.viewer.SetRenderWindow(self.window)
        self.viewer.SetupInteractor(self)
        self.viewer.SetResliceModeToAxisAligned()
        if cursor:
            self.viewer.SetResliceCursor(cursor)

        self.renderer.SetBackground(0.2, 0.2, 0.2)
        style = vtkInteractorStyleImage()
        style.handler = parent
        self.set_style(style)

        return None


    @property 
    def id(self):
        return self.orientation

    @property
    def orientation(self):
        return self.viewer.GetSliceOrientation()
    

    @orientation.setter
    def orientation(self, orientation):
        self.viewer.SetSliceOrientation(orientation)
        rep = vtk.vtkResliceCursorLineRepresentation.SafeDownCast( self.viewer.GetResliceCursorWidget().GetRepresentation())
        rep.GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(orientation)
        return None


    def load_image(self, data):
        self.viewer.SetInputData(data)
        s = self.viewer.GetSliceMin()/2 + self.viewer.GetSliceMax()/2
        self.viewer.SetSlice(int(round(s)))

        return None
    

    def set_style(self, new_style):
        style = self.GetInteractorStyle()
        if style:
            style.RemoveAllObservers()

        new_style.AutoAdjustCameraClippingRangeOn()
        new_style.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(new_style)

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
        # vtkResliceImageViewer.SLICE_ORIENTATION_YZ === 0
        # vtkResliceImageViewer.SLICE_ORIENTATION_XZ === 1
        # vtkResliceImageViewer.SLICE_ORIENTATION_XY === 2

        self.ctrl_pressed = False
        self.alt_pressed = False
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(.01)
        self.views = {}
        self.picked_point = [float('nan'),]*3
        self.picked_ijk = [float('nan'),]*3
        # three orthogonal views
        for orientation in (2,0,1):
            cursor = list(self.views.values())[0].viewer.GetResliceCursor() if len(self.views) else None
            subview = SliceViewer(parent=self, cursor=cursor)
            print(subview.viewer.SliceChangedEvent)
            subview.viewer.AddObserver(subview.viewer.SliceChangedEvent, self.reslice)
            subview.viewer.GetResliceCursorWidget().AddObserver(vtk.vtkResliceCursorWidget.ResliceAxesChangedEvent, self.axis_changed)
            subview.orientation = orientation
            subview.handler = self
            self.views[orientation] = subview
            
        # 3d view
        subview = QVTK(parent=self)
        subview.renderer.SetBackground(0.6863, 0.9333, 0.9333)
        
        style = vtkInteractorStyleTrackballCamera()
        # style.AutoAdjustCameraClippingRangeOn()
        # style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press_object)
        # style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release_object)
        # style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event_object)
        
        style.SetDefaultRenderer(subview.renderer)
        subview.SetInteractorStyle(style)
        self.views[3] = subview
        

        # put 4 subviews on a 2x2 grid
        self.gridlayout = QGridLayout(parent=self)
        self.gridlayout.setContentsMargins(0,0,0,0)
        self.gridlayout.addWidget(self.axial, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.sagittal, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.coronal, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.perspective, 1, 1, 1, 1)

        return None


    def axis_changed(self, *x):
        print(x)

    def reslice(self, *coord, ):
        print(f'coord is {coord}')



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
        return self.views[0]
    
    @property
    def coronal(self):
        return self.views[1]
    
    @property
    def axial(self):
        return self.views[2]
    
    @property
    def perspective(self):
        return self.views[3]
    
    def get_slice(self):
        return [self.views[o].viewer.GetSlice() for o in range(3)]


    # def pick(self, pos, ren, debug=True):
    #     self.picker.Pick(pos[0], pos[1], 0, ren)
    #     picked_point = [float('nan'),]*3
    #     picked_ijk = [float('nan'),]*3
    #     if self.picker.GetCellId() != -1:
    #         picked_point = self.picker.GetPickPosition()
    #         self.parent().image_data.TransformPhysicalPointToContinuousIndex(picked_point, picked_ijk)
    #         self.picked_point, self.picked_ijk = picked_point, picked_ijk
    #     if debug:
    #         try:
    #             name = self.picker.GetProp3D().GetObjectName()
    #             assert name
    #         except:
    #             name = 'change'
    #         print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.picked_point), end=' ')
    #     return None


    # async def async_pick(self, pos, ren, debug=True):
    #     self.picker.Pick(pos[0], pos[1], 0, ren)
    #     picked_point = [float('nan'),]*3
    #     picked_ijk = [float('nan'),]*3
    #     if self.picker.GetCellId() != -1:
    #         picked_point = self.picker.GetPickPosition()
    #         self.parent().image_data.TransformPhysicalPointToContinuousIndex(picked_point, picked_ijk)
    #         self.picked_point, self.picked_ijk = picked_point, picked_ijk
    #     if debug:
    #         try:
    #             name = self.picker.GetProp3D().GetObjectName()
    #             assert name
    #         except:
    #             name = 'change'
    #         print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.picked_point), end=' ')
    #     return None


    # def left_button_press_image(self, obj, event):
    #     self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
    #     if not self.ctrl_pressed:
    #         obj.OnLeftButtonDown()

    #     return None


    # def mouse_move_event_image(self, obj, event):
    #     self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
    #     if self.ctrl_pressed:
    #         self.set_slice(0, self.picked_ijk[0])
    #         self.set_slice(1, self.picked_ijk[1])
    #         self.set_slice(2, self.picked_ijk[2])
    #     else:
    #         obj.OnMouseMove()
    #     return None


    # def left_button_release_image(self, obj, event):
    #     # do not rely on clean up
    #     # handle stuff on the fly using move event and implement abort
    #     obj.OnLeftButtonUp()

    #     return None
    
    # def left_button_press_object(self, obj, event):
    #     # self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
    #     if not self.ctrl_pressed:
    #         obj.OnLeftButtonDown()

    #     return None


    # def mouse_move_event_object(self, obj, event):
    #     asy = False
    #     if asy:
    #         if self.ctrl_pressed:
    #             if hasattr(self, 'move_loop'):
    #                 try:
    #                     self.move_loop.stop()
    #                 except:
    #                     pass
    #             self.move_loop = asyncio.new_event_loop()
    #             pos, ren = obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer()
    #             self.move_loop.run_until_complete(pick_point(self, pos, ren))
    #         else:
    #             obj.OnMouseMove()
    #         return None
    #     else:
    #         if self.ctrl_pressed:
    #             self.pick(obj.GetInteractor().GetEventPosition(), obj.GetDefaultRenderer())
    #             self.set_slice(0, self.picked_ijk[0])
    #             self.set_slice(1, self.picked_ijk[1])
    #             self.set_slice(2, self.picked_ijk[2])
    #         else:
    #             obj.OnMouseMove()
    #         return None


    # def left_button_release_object(self, obj, event):
    #     # do not rely on clean up
    #     # handle stuff on the fly using move event and implement abort
    #     obj.OnLeftButtonUp()

    #     return None

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
            if event.key() == Qt.Key_A.value:
                print("A pressed")
                
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
        


class LandmarkSidePanel(QWidget):

    def __init__(self, *initargs):
        super().__init__(*initargs)

        self.tree = QTreeWidget(self)
        self.name = QLabel(self)
        self.name.setText('a')
        self.detail = QLabel(self)
        self.detail.setText('b')
        self.vtkWidget = QVTK(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tree)
        self.layout.addWidget(self.name)
        self.layout.addWidget(self.detail)
        self.layout.addWidget(self.vtkWidget)

        # self.load_lmk_tree()
        self.current_landmark = None

        self.ren = self.vtkWidget.renderer
        self.iren = self.vtkWidget.window.GetInteractor()

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
        self.ren.AddActor(actor)
        self.ren.ResetCamera()
        

    def load_lmk_tree(self):
        
        self.lib = library.Library(r'CASS.db')
        groups = {}
        for d in self.lib:
            if d.Group not in groups:
                item = QTreeWidgetItem()
                item.setText(0, d.Group)
                self.tree.addTopLevelItem(item)
                groups[d.Group] = item

            item = QTreeWidgetItem()
            item.setText(1, d.Name)
            item.setText(2, d.print_str()['Definition'])
            groups[d.Group].addChild(item)

        for x in groups.values():
            self.tree.expandItem(x)

        self.tree.itemClicked.connect(self.lmk_changed)


    def lmk_changed(self, item):
        if len(item.text(1)):
            item = self.lib.find(Name=item.text(1))
            if not item:
                return
            self.name.setText(item.Name)
            self.detail.setText(item.print_str()['Fullname'] + '\n' + item.print_str()['Description'])
            self.current_landmark = item.Name


    def show(self, *args, **kwargs):
        # self.load_lmk_tree()
        super().show(*args, **kwargs)
        self.iren.Initialize()





class AppWindow(QMainWindow, DataManager):

    def __init__(self, *args, **kw):

        # init super
        super().__init__(*args, **kw)

        # init children - main 2x2 window + side bar
        self.setWindowTitle('NOT LOADED')
        self.central = QVTK2x2Window(self)
        self.setCentralWidget(self.central)
        # lmk_panel = LandmarkSidePanel()
        # lmk_panel.show()
        # side_panel = QDockWidget()
        # side_panel.setWidget(lmk_panel)
        # self.addDockWidget(Qt.RightDockWidgetArea, side_panel)
        # side_panel.show()
        return None



    # def create_polydata(thresh, )


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



def threshold_image(image_data, thresh):
    threshold = vtkImageThreshold()
    threshold.SetInputData(image_data)
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




