
import numpy
import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList

from vtkmodules.vtkCommonTransforms import (
    vtkMatrixToLinearTransform,
    vtkLinearTransform,
    vtkTransform
)
from vtkmodules.vtkCommonDataModel import (
    vtkTriangleStrip,
    vtkPolyData,
    vtkImageData,
    vtkDataSetAttributes,
    vtkPlane,
    vtkIterativeClosestPointTransform,
    vtkCellArray,
    vtkPolyLine,
    vtkVertex
)
from vtkmodules.vtkFiltersCore import (
    vtkPolyDataNormals,
    vtkClipPolyData,
    vtkGlyph3D,
    vtkFlyingEdges3D,
    vtkSmoothPolyDataFilter,
    vtkMaskFields,
    vtkThreshold,
    vtkWindowedSincPolyDataFilter,
    vtkCleanPolyData
)
from vtkmodules.vtkFiltersGeneral import (
    vtkTransformPolyDataFilter,
    vtkTransformFilter,
    vtkDiscreteFlyingEdges3D,
    vtkDiscreteMarchingCubes
)
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)


def polydata_from_stl(file:str):
    reader = vtkSTLReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()


def polydata_from_mask(mask_image:vtkImageData, in_value=1.0, feature_angle=120.0, pass_band=0.001, smoothing_iterations=15):

    discrete_cubes = vtkDiscreteFlyingEdges3D()
    smoother = vtkWindowedSincPolyDataFilter()
    scalars_off = vtkMaskFields()
    geometry = vtkGeometryFilter()

    # closer = vtkImageOpenClose3D()
    # closer.SetInputData(label_data)
    # closer.SetKernelSize(2, 2, 2)
    # closer.SetCloseValue(in_value)

    discrete_cubes.SetInputData(mask_image)
    discrete_cubes.SetValue(0, in_value)
    
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


def polydata_from_points(pts:vtkPoints, sphere_radius=1.0, return_glyph=False):

    input = vtkPolyData()
    input.SetPoints(pts)

    glyphSource = vtkSphereSource()
    glyphSource.SetRadius(sphere_radius)
    glyphSource.Update()

    glyph3D = vtkGlyph3D()
    glyph3D.GeneratePointIdsOn()
    glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
    glyph3D.SetInputData(input)
    glyph3D.SetScaleModeToDataScalingOff()
    glyph3D.Update()
    if return_glyph:
        return glyph3D.GetOutput(), glyph3D
    return glyph3D.GetOutput()



def write_polydata_to_stl(polyd:vtkPolyData, file:str):
    writer = vtkSTLWriter()
    writer.SetInputData(polyd)
    writer.SetFileName(file)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def clean_polydata(polyd:vtkPolyData):
    cleaner = vtkCleanPolyData()
    cleaner.SetInputData(polyd)
    cleaner.Update()
    return cleaner.GetOutput()


def translate_polydata(polyd:vtkPolyData, t:numpy.ndarray):
    T = vtkTransform()
    T.Translate(t.ravel())
    Transform = vtkTransformPolyDataFilter()
    Transform.SetTransform(T)
    Transform.SetInputData(polyd)
    Transform.Update()
    return Transform.GetOutput()


def transform_polydata(polyd:vtkPolyData, t:numpy.ndarray):
    T = vtkTransform()
    M = vtkMatrix4x4()
    M.DeepCopy(t.ravel())
    T.SetMatrix(M)
    Transform = vtkTransformPolyDataFilter()
    Transform.SetTransform(T)
    Transform.SetInputData(polyd)
    Transform.Update()
    return Transform.GetOutput()


def flip_normals_polydata(polyd:vtkPolyData):
    T = vtkPolyDataNormals()
    T.FlipNormalsOn()
    T.SetInputData(polyd)
    T.Update()
    return T.GetOutput()


def cut_polydata(polyd:vtkPolyData, plane_origin:numpy.ndarray, plane_normal:numpy.ndarray):
    plane = vtkPlane()
    plane.SetNormal(plane_normal)
    plane.SetOrigin(plane_origin)

    cutter = vtkClipPolyData()
    cutter.SetClipFunction(plane)
    cutter.SetInputData(polyd)
    cutter.Update()

    return cutter.GetOutput()


def icp_register(source, target):

    T = vtkIterativeClosestPointTransform()
    T.SetSource(source)
    T.SetTarget(target)
    T.Update()
    return T.GetOutput()


def triangle_strip(ind_1d:numpy.ndarray):
    l = vtkTriangleStrip()
    l.GetPointIds().SetNumberOfIds(ind_1d.size)
    for i,k in enumerate(ind_1d):
        l.GetPointIds().SetId(i,k)
    strip = vtkCellArray()
    strip.InsertNextCell(l)
    return strip

def polylines(ind_2d:numpy.ndarray): # each row is a polyline
    lines = vtkCellArray()

    for i, id in enumerate(ind_2d):

        l = vtkPolyLine()
        l.GetPointIds().SetNumberOfIds(id.size)

        for i,k in enumerate(id):
            l.GetPointIds().SetId(i,k)

        lines.InsertNextCell(l)
        return lines

def test():
    pass


if __name__=='__main__':
    test()


