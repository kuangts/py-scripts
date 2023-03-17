#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkImagingCore import vtkExtractVOI
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkDataSetMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkImagingCore import vtkImageReslice, vtkImagePermute
from vtkmodules.vtkCommonMath import vtkMatrix4x4
import vtk

def main():
    fileName = r'C:\data\pre-post-paired-40-send-1122\n0001\20110425-pre.nii.gz'

    colors = vtkNamedColors()

    ren_win = vtkRenderWindow()
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    reader = vtkNIFTIImageReader()
    reader.SetFileName(fileName)
    reader.Update()

    permute_axial = vtkImagePermute()
    permute_axial.SetInputConnection(reader.GetOutputPort())
    permute_axial.SetFilteredAxes(0,1,2)

    permute_sagittal = vtkImagePermute()
    permute_sagittal.SetInputConnection(reader.GetOutputPort())
    permute_sagittal.SetFilteredAxes(1,2,0)

    permute_coronal = vtkImagePermute()
    permute_coronal.SetInputConnection(reader.GetOutputPort())
    permute_coronal.SetFilteredAxes(0,2,1)

    extract_axial = vtkExtractVOI()
    extract_axial.SetInputConnection(permute_axial.GetOutputPort())
    extract_axial.SetVOI(0,511,0,511,100,100)
    extract_axial.Update()

    extract_sagittal = vtkExtractVOI()
    extract_sagittal.SetInputConnection(permute_sagittal.GetOutputPort())
    extract_sagittal.SetVOI(0,511,0,143,200,200)
    extract_sagittal.Update()

    extract_coronal = vtkExtractVOI()
    extract_coronal.SetInputConnection(permute_coronal.GetOutputPort())
    extract_coronal.SetVOI(0,511,0,143,300,300)
    extract_coronal.Update()

    # Create a greyscale lookup table
    table = vtk.vtkLookupTable()
    table.SetRange(0, 2000) # image intensity range
    table.SetValueRange(0.0, 1.0) # from black to white
    table.SetSaturationRange(0.0, 0.0) # no color saturation
    table.SetRampToLinear()
    table.Build()

    # Map the image through the lookup table
    color_axial = vtk.vtkImageMapToColors()
    color_axial.SetLookupTable(table)
    color_axial.SetInputConnection(extract_axial.GetOutputPort())

    color_sagittal = vtk.vtkImageMapToColors()
    color_sagittal.SetLookupTable(table)
    color_sagittal.SetInputConnection(extract_sagittal.GetOutputPort())

    color_coronal = vtk.vtkImageMapToColors()
    color_coronal.SetLookupTable(table)
    color_coronal.SetInputConnection(extract_coronal.GetOutputPort())

    img_axial = vtk.vtkImageActor()
    img_axial.GetMapper().SetInputConnection(color_axial.GetOutputPort())
    img_axial.GetMapper().Update()

    img_sagittal = vtk.vtkImageActor()
    img_sagittal.GetMapper().SetInputConnection(color_sagittal.GetOutputPort())
    img_sagittal.GetMapper().Update()

    img_coronal = vtk.vtkImageActor()
    img_coronal.GetMapper().SetInputConnection(color_coronal.GetOutputPort())
    img_coronal.GetMapper().Update()



    # Display the image

    # Have some fun with colors.
    ren_bkg = ['AliceBlue', 'GhostWhite', 'WhiteSmoke', 'Seashell']
    actor_color = ['Bisque', 'RosyBrown', 'Goldenrod', 'Chocolate']

    ren_axial = vtkRenderer()
    ren_axial.AddActor(img_axial)
    ren_axial.SetBackground(colors.GetColor3d('SlateGray'))
    ren_axial.ResetCamera()
    # ren_axial.GetActiveCamera().Dolly(1.5)
    # ren_axial.ResetCameraClippingRange()
    ren_win.AddRenderer(ren_axial)
    ren_axial.SetViewport(0,.5,.5,1)

    ren_sagittal = vtkRenderer()
    ren_sagittal.AddActor(img_sagittal)
    ren_sagittal.SetBackground(colors.GetColor3d('SlateGray'))
    ren_sagittal.ResetCamera()
    # ren_sagittal.GetActiveCamera().Dolly(1.5)
    # ren_sagittal.ResetCameraClippingRange()
    ren_win.AddRenderer(ren_sagittal)
    ren_sagittal.SetViewport(.5,.5,1,1)

    ren_coronal = vtkRenderer()
    ren_coronal.AddActor(img_coronal)
    ren_coronal.SetBackground(colors.GetColor3d('SlateGray'))
    ren_coronal.ResetCamera()
    # ren_coronal.GetActiveCamera().Dolly(1.5)
    # ren_coronal.ResetCameraClippingRange()
    ren_win.AddRenderer(ren_coronal)
    ren_coronal.SetViewport(0,0, .5,.5)

    # style.SetDefaultRenderer(ren_axial)
    
    iren.Initialize()
    ren_win.Render()
    iren.Start()




if __name__ == '__main__':
    main()