#!/usr/bin/env python

import os, sys
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOImport import vtk3DSImporter
from vtkmodules.vtkRenderingCore import (
    vtkCamera,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter

def main(argv):
    fileName = argv[1]
    importer = vtk3DSImporter()
    importer.SetFileName(fileName)
    importer.ComputeNormalsOn()


    renderer = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(renderer)
    importer.SetRenderWindow(renWin)
    importer.Update()
    renWin.SetWindowName(fileName)

    camera = vtkCamera()
    camera.SetPosition(0, -1, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    camera.Azimuth(0)
    camera.Elevation(0)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()

    renWin.Render()

    names = ['di','diL','diR','le']
    actors = renderer.GetActors()
    if actors.GetNumberOfItems() == 5:
        names.append('gen')
        names = ['pre_'+s for s in names]
        names.sort(reverse=True)
    if actors.GetNumberOfItems() == 6:
        names.append('gen')
        names = ['pre_'+s for s in names]
        names.append('hex_bone')
        names.sort(reverse=True)

    for i in range(actors.GetNumberOfItems()):
        actor = actors.GetItemAsObject(i)
        polyd = actor.GetMapper().GetInput()

        writer = vtkSTLWriter()
        writer.SetInputData(polyd)
        writer.SetFileName(os.path.join(os.path.dirname(fileName),rf'{names[i]}.stl'))
        writer.SetFileTypeToBinary()
        writer.Update()
        writer.Write()


    renWin.Finalize()




if __name__ == '__main__':

    main(sys.argv)