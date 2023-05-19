# vtkRegularPolygonSource defining regions for clipping/cutting
# 2d polygon boolean
# vtkstripper
# ray cast from boundary to mandible
import glob, os, csv, shutil, re
from os.path import join as pjoin
from os.path import exists as pexists
from os.path import isfile as isfile
from os.path import isdir as isdir
from os.path import basename, dirname, normpath, realpath
from typing import Any
from vtk import vtkRegularPolygonSource, vtkPolygon

import numpy as np
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkParametricFunctionSource
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkTriangleFilter, vtkClipPolyData, vtkPolyDataConnectivityFilter, vtkImplicitPolyDataDistance, vtkAppendPolyData
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkPolyLine, vtkUnstructuredGrid, vtkImplicitSelectionLoop, vtkPointLocator, vtkImplicitDataSet, vtkPolyDataCollection
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkFloatArray
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget, vtkBoxRepresentation
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkTransformFilter, vtkBooleanOperationPolyDataFilter 
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
    vtkRenderer,
    vtkProp3DFollower,
    vtkCoordinate
)
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from vtkmodules.vtkCommonComputationalGeometry import vtkParametricSpline
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkPolyDataNormals
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkSmoothPolyDataFilter, vtkCleanPolyData, vtkFeatureEdges, vtkStripper
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkDiscreteFlyingEdges3D, vtkTransformFilter
from vtkmodules.vtkIOImage import vtkNIFTIImageReader, vtkNIFTIImageHeader
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
    vtkDataSetAttributes,
    vtkIterativeClosestPointTransform,
    vtkPlane,
    vtkBox
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
from vtkmodules.vtkInteractionWidgets import vtkPolygonalSurfacePointPlacer, vtkContourWidget, vtkOrientedGlyphContourRepresentation
from vtkmodules.vtkFiltersModeling import vtkSelectPolyData, vtkRibbonFilter
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData
)
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter



  

def main():

  colors = vtkNamedColors()

  sphereSource = vtkSphereSource()
  sphereSource.SetRadius(5)
  sphereSource.Update()

  mapper = vtkPolyDataMapper()
  mapper.SetInputConnection(sphereSource.GetOutputPort())

  actor = vtkActor()
  actor.SetMapper(mapper)
  actor.GetProperty().SetColor(colors.GetColor3d("MistyRose"))

  renderer = vtkRenderer()
  renderWindow = vtkRenderWindow()
  renderWindow.AddRenderer(renderer)
  renderWindow.SetWindowName("PolygonalSurfacePointPlacer")

  interactor = vtkRenderWindowInteractor()
  interactor.SetRenderWindow(renderWindow)

  renderer.AddActor(actor)
  renderer.SetBackground(colors.GetColor3d("CadetBlue"))

  contourWidget = vtkContourWidget()
  contourWidget.SetInteractor(interactor)

  contourWidget.AddObserver('InteractionEvent', lambda x,*_: x.GetRepresentation().GetNumberOfNodes())

  rep = contourWidget.GetRepresentation()

  pointPlacer = vtkPolygonalSurfacePointPlacer()
  pointPlacer.AddProp(actor)
#   pointPlacer.GetPolys().AddItem(sphereSource.GetOutput())

  rep.GetLinesProperty().SetColor(colors.GetColor3d("Crimson"))
  rep.GetLinesProperty().SetLineWidth(3.0)
  rep.SetPointPlacer(pointPlacer)

  contourWidget.EnabledOn()
  contourWidget.GetRepresentation().AlwaysOnTopOn()
  renderer.ResetCamera()
  renderWindow.Render()
  interactor.Initialize()

  interactor.Start()


if __name__ == '__main__':
  main()