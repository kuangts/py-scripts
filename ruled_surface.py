#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData,
    vtkPolyLine
)
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def main():
    colors = vtkNamedColors()

    # Create a rendering window and renderer
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the points fot the lines.
    points = vtkPoints()
    points.InsertPoint(0, 1, 0, 0.1)
    points.InsertPoint(1, 1, 1, 0.1)
    points.InsertPoint(2, 2, 0, 0.2)
    points.InsertPoint(3, 2, 1, 0.3)
    points.InsertPoint(4, 3, 0, 0.3)
    points.InsertPoint(5, 3, 1, 0.2)
    points.InsertPoint(6, 4, 0, 0.2)
    points.InsertPoint(7, 4, 1, 0.1)

    # Create line1
    line1 = vtkPolyLine()
    line1.GetPointIds().SetNumberOfIds(4)
    line1.GetPointIds().SetId(0, 0)
    line1.GetPointIds().SetId(1, 2)
    line1.GetPointIds().SetId(2, 4)
    line1.GetPointIds().SetId(3, 6)

    # Create line2
    line2 = vtkPolyLine()
    line2.GetPointIds().SetNumberOfIds(4)
    line2.GetPointIds().SetId(0, 1)
    line2.GetPointIds().SetId(1, 3)
    line2.GetPointIds().SetId(2, 5)
    line2.GetPointIds().SetId(3, 7)

    # Create a cellArray containing the lines
    lines = vtkCellArray()
    lines.InsertNextCell(line1)
    lines.InsertNextCell(line2)

    # Create the vtkPolyData to contain the points and cellArray with the lines
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Create the ruledSurfaceFilter from the polydata containing the lines
    ruledSurfaceFilter = vtkRuledSurfaceFilter()
    ruledSurfaceFilter.SetInputData(polydata)
    ruledSurfaceFilter.SetResolution(21, 21)
    ruledSurfaceFilter.SetRuledModeToResample()

    # Create the mapper with the ruledSurfaceFilter as input
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(ruledSurfaceFilter.GetOutputPort())

    # Create the actor with the mapper
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
    actor.GetProperty().SetSpecular(0.6)
    actor.GetProperty().SetSpecularPower(30)

    # Add the actor to the display
    ren.AddActor(actor)
    ren.SetBackground(colors.GetColor3d("SteelBlue"))
    renWin.SetWindowName('RuledSurfaceFilter')

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    # ren.GetActiveCamera().SetPosition(3.7, -0.5, -0.5)
    # ren.GetActiveCamera().SetFocalPoint(0.5, 0.5, 0.5)
    # ren.GetActiveCamera().SetViewUp(-0.3, 0.1, -1.0)
    ren.GetActiveCamera().Azimuth(60)
    ren.GetActiveCamera().Elevation(60)
    ren.ResetCamera()

    renWin.Render()
    iren.Start()


if __name__ == '__main__':
    main()