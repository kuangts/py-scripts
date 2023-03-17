#!/usr/bin/env python

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
    vtkRenderer
)
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkImagingCore import vtkExtractVOI
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkImagingCore import vtkImageReslice, vtkImagePermute
import vtk


class vtk2x2RenderWindow(vtkRenderWindow):
        
    # def __init__(self):
    #     pass


    def setup(self, filename=None, **kw):

        # One render window, multiple viewports.
        super().__init__(**kw)

        # Define viewport ranges.
        xmins = [0, 0, .5]
        xmaxs = [0.5, 0.5, 1]
        ymins = [0, .5, .5]
        ymaxs = [0.5, 1, 1]
        filtax = [0, 0, 1]
        filtay = [2, 1, 2]
        filtaz = [1, 2, 0]

        ren_list = []

        self.reader = vtkNIFTIImageReader()
        if filename is not None:
            self.reader.SetFileName(filename)
            self.reader.Update()


        # Create a greyscale lookup table
        table = vtk.vtkLookupTable()
        table.SetRange(0, 2000) # image intensity range
        table.SetValueRange(0.0, 1.0) # from black to white
        table.SetSaturationRange(0.0, 0.0) # no color saturation
        table.SetRampToLinear()
        table.Build()

        for i in range(3):
            ren = vtkRenderer()
            self.AddRenderer(ren)
            ren.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

            permute = vtkImagePermute()
            permute.SetInputConnection(self.reader.GetOutputPort())
            permute.SetFilteredAxes(filtax[i], filtay[i], filtaz[i])

            permute.Update()
            extent = list(permute.GetOutput().GetExtent())
            extent[4] = extent[5] // 2
            extent[5] = extent[4]

            extract = vtkExtractVOI()
            extract.SetInputConnection(permute.GetOutputPort())
            extract.SetVOI(*extent)
            extract.Update()

            # Map the image through the lookup table
            color = vtk.vtkImageMapToColors()
            color.SetLookupTable(table)
            color.SetInputConnection(extract.GetOutputPort())

            img = vtk.vtkImageActor()
            img.GetMapper().SetInputConnection(color.GetOutputPort())

            ren.AddActor(img)
            ren.SetBackground(0.44, 0.50, 0.56)

            ren.ResetCamera()
            ren_list.append(ren)




if __name__ == '__main__':
    colors = vtkNamedColors()
    # Have some fun with colors.
    ren_bkg = ['AliceBlue', 'GhostWhite', 'WhiteSmoke', 'Seashell']
    actor_color = ['Bisque', 'RosyBrown', 'Goldenrod', 'Chocolate']


    rw = vtk2x2RenderWindow()
    rw.setup(filename=r'C:\data\pre-post-paired-40-send-1122\n0001\20110425-pre.nii.gz')

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)

    style = vtkInteractorStyleImage()
    iren.SetInteractorStyle(style)

    iren.Initialize()
    rw.Render()
    iren.Start()

