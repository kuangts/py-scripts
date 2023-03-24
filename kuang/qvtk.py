#!/usr/bin/env python
## python packages
import sys, os, glob, json

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
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkLookupTable
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
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import (
    vtkConeSource,
    vtkCubeSource,
    vtkCylinderSource,
    vtkSphereSource
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkImageActor
)

from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkImagingCore import vtkExtractVOI, vtkImageMapToColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkImagingCore import vtkImagePermute

ren_bkg = ['AliceBlue', 'GhostWhite', 'WhiteSmoke', 'Seashell']
actor_color = ['Bisque', 'RosyBrown', 'Goldenrod', 'Chocolate']

bkg = (0,0,0)

class MyApplication(QApplication):
    windows = []
    def new_window(self):
        w = AppWindow()
        self.windows.append(w)
        w.show()
        return w

class AppWindow(QMainWindow):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.resize(640,480)
        self.setWindowTitle('NOTHING IS LOADED')
        central = vtk2x2RenderWindow()
        vtkWidget = QVTKRenderWindowInteractor(self, rw=central)
        style = vtkInteractorStyleImage()
        self.iren = vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(style)
        style.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        self.setCentralWidget(vtkWidget)
        return None

    def left_button_press_event(self, obj, event):
        pos = self.iren.GetEventPosition()
        print(pos)

    def load_data(self, **kw):
        if 'nifti_file' in kw:
            self.load_nifti(kw['nifti_file'])

        return None
    
    def load_nifti(self, file):
        src = vtkNIFTIImageReader()
        src.SetFileName(file)
        src.Update()
        self.centralWidget().GetRenderWindow().set_source(src)
        self.setWindowTitle('VIEWERING: ' + file)
        return None

class vtk2x2RenderWindow(vtkRenderWindow):
        
    def __init__(self, source=None):

        self.loaded = False
        super().__init__()

        # Create a greyscale lookup table
        self.grayscale_table = vtkLookupTable()
        self.grayscale_table.SetRange(0, 2000) # image intensity range
        self.grayscale_table.SetValueRange(0.0, 1.0) # from bkg to white
        self.grayscale_table.SetSaturationRange(0.0, 0.0) # no color saturation
        self.grayscale_table.SetRampToLinear()
        self.grayscale_table.Build()

        if source:
            self.set_source(source)

        return None

    def set_source(self, src):

        self.source = src

        if self.loaded:
            for i in range(3):
                self.permutors[i].SetInputConnection(src.GetOutputPort())
                self.permutors[i].Update()
                extent = list(self.permutors[i].GetOutput().GetExtent())
                extent[4] = (extent[4]+extent[5])//2
                extent[5] = extent[4]
                self.extractors[i].SetVOI(*extent)
                self.extractors[i].Update()
        else:

            extent = self.source.GetOutput().GetExtent()
            
            permute_axial = vtkImagePermute()
            permute_axial.SetFilteredAxes(0,1,2)
            permute_axial.SetInputConnection(src.GetOutputPort())
            extract_axial = vtkExtractVOI()
            extract_axial.SetInputConnection(permute_axial.GetOutputPort())
            extract_axial.SetVOI(*extent[0:4],(extent[4]+extent[5])//2,(extent[4]+extent[5])//2)
            color_axial = vtkImageMapToColors()
            color_axial.SetLookupTable(self.grayscale_table)
            color_axial.SetInputConnection(extract_axial.GetOutputPort())
            img_axial = vtkImageActor()
            img_axial.GetMapper().SetInputConnection(color_axial.GetOutputPort())
            img_axial.GetMapper().Update()
            ren_axial = vtkRenderer()
            ren_axial.SetViewport(0,.5,.5,1)
            ren_axial.AddActor(img_axial)
            ren_axial.SetBackground(bkg)
            ren_axial.ResetCamera()
            ren_axial.GetActiveCamera().Dolly(1.5)
            ren_axial.ResetCameraClippingRange()
            self.AddRenderer(ren_axial)

            permute_sagittal = vtkImagePermute()
            permute_sagittal.SetFilteredAxes(1,2,0)
            permute_sagittal.SetInputConnection(src.GetOutputPort())
            extract_sagittal = vtkExtractVOI()
            extract_sagittal.SetInputConnection(permute_sagittal.GetOutputPort())
            extract_sagittal.SetVOI(*extent[2:6],(extent[0]+extent[1])//2,(extent[0]+extent[1])//2)
            color_sagittal = vtkImageMapToColors()
            color_sagittal.SetLookupTable(self.grayscale_table)
            color_sagittal.SetInputConnection(extract_sagittal.GetOutputPort())
            img_sagittal = vtkImageActor()
            img_sagittal.GetMapper().SetInputConnection(color_sagittal.GetOutputPort())
            img_sagittal.GetMapper().Update()
            ren_sagittal = vtkRenderer()
            ren_sagittal.SetViewport(.5,.5,1,1)
            ren_sagittal.AddActor(img_sagittal)
            ren_sagittal.SetBackground(bkg)
            ren_sagittal.ResetCamera()
            ren_sagittal.GetActiveCamera().Dolly(1.5)
            ren_sagittal.ResetCameraClippingRange()
            self.AddRenderer(ren_sagittal)

            permute_coronal = vtkImagePermute()
            permute_coronal.SetFilteredAxes(0,2,1)
            permute_coronal.SetInputConnection(src.GetOutputPort())
            extract_coronal = vtkExtractVOI()
            extract_coronal.SetInputConnection(permute_coronal.GetOutputPort())
            extract_coronal.SetVOI(*extent[0:2], *extent[4:6], (extent[2]+extent[3])//2,(extent[2]+extent[3])//2)
            color_coronal = vtkImageMapToColors()
            color_coronal.SetLookupTable(self.grayscale_table)
            color_coronal.SetInputConnection(extract_coronal.GetOutputPort())
            img_coronal = vtkImageActor()
            img_coronal.GetMapper().SetInputConnection(color_coronal.GetOutputPort())
            img_coronal.GetMapper().Update()
            ren_coronal = vtkRenderer()
            ren_coronal.SetViewport(0, 0, .5,.5)
            ren_coronal.AddActor(img_coronal)
            ren_coronal.SetBackground(bkg)
            ren_coronal.ResetCamera()
            ren_coronal.GetActiveCamera().Dolly(1.5)
            ren_coronal.ResetCameraClippingRange()
            self.AddRenderer(ren_coronal)

            self.renderer_axial = ren_axial
            self.renderer_sagittal = ren_sagittal
            self.renderer_coronal = ren_coronal

            self.image_axial = img_axial
            self.image_sagittal = img_sagittal
            self.image_coronal = img_coronal

            self.extractors = [extract_axial, extract_sagittal, extract_coronal]
            self.permutors = [permute_axial, permute_sagittal, permute_coronal]
            
            self.loaded = True
        
        self.Render()
        return None

def main(argv):
    app = MyApplication(argv)

    w = app.new_window()
    w.load_nifti(r'C:\data\pre-post-paired-40-send-1122\n0001\20110425-pre.nii.gz')
    w.load_data(nifti_file=r'C:\data\pre-post-paired-40-send-1122\n0001\20110425-pre.nii.gz')

    # w.centralWidget().GetRenderWindow().renderer_axial.SetViewport(0,0,1,1)
    
    sys.exit(app.exec())



if __name__ == "__main__":

    main(sys.argv)



