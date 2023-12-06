import sys, os, glob, json, math
from abc import abstractmethod, ABC
from collections.abc import Collection
## site packages
import numpy as np
from scipy.spatial  import KDTree
from numbers import Number
# vtk
import vtk
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkImageData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget, vtkResliceCursorWidget, vtkResliceCursorLineRepresentation, vtkResliceCursor
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



class SliceViewer(QVTKRenderWindowInteractor):
    
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
        style.AutoAdjustCameraClippingRangeOn()
        style.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)

        return None



    def load_image(self, data):
        self.viewer.SetInputData(data)
        s = self.viewer.GetSliceMin()/2 + self.viewer.GetSliceMax()/2
        self.viewer.SetSlice(int(round(s)))

        return None
    

# class KeyFilter(QObject):
#     def eventFilter(self, obj, event):
#         if isinstance(event, QKeyEvent):
#             obj.parent().event(event)
#             return True
#         else:
#             # standard event processing
#             return super().eventFilter(obj, event)


AXIAL = vtkResliceImageViewer.SLICE_ORIENTATION_XY
CORONAL = vtkResliceImageViewer.SLICE_ORIENTATION_XZ
SAGITTAL = vtkResliceImageViewer.SLICE_ORIENTATION_YZ

class QVTK2x2Window(QWidget):
    # this class is where all vtk events are handled
    # mode is controled by its parent on view hierarchy thru delegation
    # 
    reslice_delegate = Signal(float, float, float)
    def __init__(self, *initargs):
        super().__init__(*initargs)
  
        self.image_picker = vtkCellPicker()
        self.image_picker.SetTolerance(.01)
        # three orthogonal views
        
        # sagittal
        self.iren_sagittal = QVTKRenderWindowInteractor()
        self.viewer_sagittal = vtkResliceImageViewer()
        style = vtkInteractorStyleImage()
        self.iren_sagittal.SetInteractorStyle(style)
        self.iren_sagittal.GetRenderWindow().AddRenderer(self.viewer_sagittal.GetRenderer())
        self.viewer_sagittal.SetRenderWindow(self.iren_sagittal.GetRenderWindow())
        self.viewer_sagittal.SetupInteractor(self.iren_sagittal)
        self.viewer_sagittal.GetInteractorStyle().AddObserver('LeftButtonPressEvent', self.left_button_press_event_image)
        # self.viewer_sagittal.GetInteractorStyle().AddObserver('MouseMoveEvent', self.left_button_move_event_image)
        self.viewer_sagittal.GetInteractorStyle().AutoAdjustCameraClippingRangeOn()
        self.viewer_sagittal.GetInteractorStyle().SetDefaultRenderer(self.viewer_sagittal.GetRenderer())
        self.viewer_sagittal.SetResliceModeToAxisAligned()
        self.viewer_sagittal.SetSliceOrientation(vtkResliceImageViewer.SLICE_ORIENTATION_YZ)
        rep = vtk.vtkResliceCursorLineRepresentation.SafeDownCast( self.viewer_sagittal.GetResliceCursorWidget().GetRepresentation())
        rep.GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(vtkResliceImageViewer.SLICE_ORIENTATION_YZ)
        # axial
        self.iren_axial = QVTKRenderWindowInteractor()
        self.viewer_axial = vtkResliceImageViewer()
        style = vtkInteractorStyleImage()
        self.iren_axial.SetInteractorStyle(style)
        self.iren_axial.GetRenderWindow().AddRenderer(self.viewer_axial.GetRenderer())
        self.viewer_axial.SetRenderWindow(self.iren_axial.GetRenderWindow())
        self.viewer_axial.SetupInteractor(self.iren_axial)
        self.viewer_axial.GetInteractorStyle().AddObserver('LeftButtonPressEvent', self.left_button_press_event_image)
        # self.viewer_axial.GetInteractorStyle().AddObserver('MouseMoveEvent', self.left_button_move_event_image)
        self.viewer_axial.GetInteractorStyle().AutoAdjustCameraClippingRangeOn()
        self.viewer_axial.GetInteractorStyle().SetDefaultRenderer(self.viewer_axial.GetRenderer())
        self.viewer_axial.SetResliceModeToAxisAligned()
        self.viewer_axial.SetSliceOrientation(vtkResliceImageViewer.SLICE_ORIENTATION_XY)
        rep = vtk.vtkResliceCursorLineRepresentation.SafeDownCast( self.viewer_axial.GetResliceCursorWidget().GetRepresentation())
        rep.GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(vtkResliceImageViewer.SLICE_ORIENTATION_XY)
        # coronal
        self.iren_coronal = QVTKRenderWindowInteractor()
        self.viewer_coronal = vtkResliceImageViewer()
        style = vtkInteractorStyleImage()
        self.iren_coronal.SetInteractorStyle(style)
        self.iren_coronal.GetRenderWindow().AddRenderer(self.viewer_coronal.GetRenderer())
        self.viewer_coronal.SetRenderWindow(self.iren_coronal.GetRenderWindow())
        self.viewer_coronal.SetupInteractor(self.iren_coronal)
        self.viewer_coronal.GetInteractorStyle().AddObserver('LeftButtonPressEvent', self.left_button_press_event_image)
        # self.viewer_coronal.GetInteractorStyle().AddObserver('MouseMoveEvent', self.left_button_move_event_image)
        self.viewer_coronal.GetInteractorStyle().AutoAdjustCameraClippingRangeOn()
        self.viewer_coronal.GetInteractorStyle().SetDefaultRenderer(self.viewer_coronal.GetRenderer())
        self.viewer_coronal.SetResliceModeToAxisAligned()
        self.viewer_coronal.SetSliceOrientation(vtkResliceImageViewer.SLICE_ORIENTATION_XZ)
        rep = vtk.vtkResliceCursorLineRepresentation.SafeDownCast( self.viewer_coronal.GetResliceCursorWidget().GetRepresentation())
        rep.GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(vtkResliceImageViewer.SLICE_ORIENTATION_XZ)
        # 3d view
        self.iren_3d = QVTKRenderWindowInteractor(parent=self)
        self.renderer_3d = vtkRenderer()
        self.renderer_3d.SetBackground(0.6863, 0.9333, 0.9333)
        self.iren_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        style = vtkInteractorStyleTrackballCamera()
        style.AutoAdjustCameraClippingRangeOn()
        style.SetDefaultRenderer(self.renderer_3d)
        self.iren_3d.SetInteractorStyle(style)

        cursor = self.viewer_axial.GetResliceCursor()
        self.viewer_coronal.SetResliceCursor(cursor)
        self.viewer_sagittal.SetResliceCursor(cursor)



        # self.viewer_axial.AddObserver(self.viewer_axial.SliceChangedEvent, self.reslice)
        # self.viewer_coronal.AddObserver(self.viewer_coronal.SliceChangedEvent, self.reslice)
        # self.viewer_sagittal.AddObserver(self.viewer_sagittal.SliceChangedEvent, self.reslice)

        # self.viewer_axial.GetResliceCursorWidget().AddObserver(vtkResliceCursorWidget.ResliceAxesChangedEvent, self.axis_changed)
        # self.viewer_coronal.GetResliceCursorWidget().AddObserver(vtkResliceCursorWidget.ResliceAxesChangedEvent, self.axis_changed)
        # self.viewer_sagittal.GetResliceCursorWidget().AddObserver(vtkResliceCursorWidget.ResliceAxesChangedEvent, self.axis_changed)


        self.update_image(vtkImageData()) # this is to silent error from vtk before image is loaded
        self.iren_sagittal.Initialize()
        self.iren_sagittal.GetRenderWindow().Render()
        self.iren_sagittal.Start()
        self.iren_axial.Initialize()
        self.iren_axial.GetRenderWindow().Render()
        self.iren_axial.Start()
        self.iren_coronal.Initialize()
        self.iren_coronal.GetRenderWindow().Render()
        self.iren_coronal.Start()
        self.iren_3d.Initialize()
        self.iren_3d.GetRenderWindow().Render()
        self.iren_3d.Start()

        # for orientation in (2,0,1):
        #     cursor = list(self.views.values())[0].viewer.GetResliceCursor() if len(self.views) else None
            # subview = SliceViewer(parent=self, cursor=cursor)
            # print(subview.viewer.SliceChangedEvent)
            # subview.viewer.AddObserver(subview.viewer.SliceChangedEvent, self.reslice)
            # subview.viewer.GetResliceCursorWidget().AddObserver(vtk.vtkResliceCursorWidget.ResliceAxesChangedEvent, self.axis_changed)
            # subview.orientation = orientation
            # subview.handler = self
            # self.views[orientation] = subview
        
        # put 4 subviews on a 2x2 grid
        self.gridlayout = QGridLayout(parent=self)
        self.gridlayout.setContentsMargins(0,0,0,0)
        self.gridlayout.addWidget(self.iren_axial, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.iren_sagittal, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.iren_coronal, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.iren_3d, 1, 1, 1, 1)
        return None
    

    def show(self):
        self.iren_sagittal.show()
        self.iren_axial.show()
        self.iren_coronal.show()
        self.iren_3d.show()
        super().show()
        return None
    

    def update_image(self, img:vtkImageData):
        self.viewer_sagittal.SetInputData(img)
        self.viewer_axial.SetInputData(img)
        self.viewer_coronal.SetInputData(img)
        self.reslice_delegate.emit(*img.GetCenter())
        return None
    

    def left_button_press_event_image(self, obj:vtkInteractorStyleImage, event):
        pos = obj.GetInteractor().GetEventPosition()
        self.image_picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
        if self.image_picker.GetCellId() >= 0:
            self.reslice_delegate.emit(*self.image_picker.GetPickPosition())


    # def left_button_move_event_image(self, obj:vtkInteractorStyleImage, event):
    #     pos = obj.GetInteractor().GetEventPosition()
    #     self.image_picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
    #     if self.image_picker.GetCellId() >= 0:
    #         self.reslice_delegate.emit(self.image_picker.GetPickPosition())


class AppWindow(QMainWindow):
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)
        self.setWindowTitle('NOT LOADED')
        self.central = QVTK2x2Window(self)
        self.image = vtkImageData()
        self.setCentralWidget(self.central)
        self.central.reslice_delegate.connect(self.set_reslice_center)
        return None
    

    def show(self):
        self.central.show()
        super().show()
        return None
    

    def read_image(self, image_path:str):
        reader = vtkNIFTIImageReader()
        reader.SetFileName(image_path)
        reader.Update()
        self.image = reader.GetOutput()
        self.update_image(self.image)
        return None


    def update_image(self, img:vtkImageData):
        for v in self.children():
            if hasattr(v, 'update_image'):
                v.update_image(img)
        return None


    @Slot(float, float, float)
    def set_reslice_center(self, x,y,z):

        # three views must be resliced at the same time
        new_slice = [float('nan')]*3
        self.image.TransformPhysicalPointToContinuousIndex([x,y,z], new_slice)
        i,j,k = new_slice
        if i>=self.central.viewer_sagittal.GetSliceMin() and i<=self.central.viewer_sagittal.GetSliceMax():
            self.central.viewer_sagittal.SetSlice(int(round(i)))
        if j>=self.central.viewer_coronal.GetSliceMin() and j<=self.central.viewer_coronal.GetSliceMax():
            self.central.viewer_coronal.SetSlice(int(round(j)))
        if k>=self.central.viewer_axial.GetSliceMin() and k<=self.central.viewer_axial.GetSliceMax():
            self.central.viewer_axial.SetSlice(int(round(k)))

        return None



class SkullEngineClient(QApplication):
    pass


def main():
    app = SkullEngineClient([])
    app.windows = []
    w = AppWindow()
    app.windows.append(w)
    w.read_image(r'C:\Users\tians\test.nii.gz')
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':

    main()




