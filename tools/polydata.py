
import numpy
import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList, vtkAbstractArray
from vtkmodules.vtkInteractionWidgets import vtkImplicitPlaneRepresentation

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
    vtkVertex,
    vtkBox,
    vtkPolygon
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
    vtkCleanPolyData,
    vtkExtractEdges,
    vtkImplicitPolyDataDistance,
    vtkConnectivityFilter
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
from vtkmodules.util.vtkConstants import VTK_DOUBLE

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


def polydata_from_numpy(nodes, polys, lines=True, verts=False):

    # shape for the input variables
    # nodes (_,3)
    # polys (_,x)
    # lines (_,2)
    # nodes (_,1) or (_,)

    N = vtkPoints()
    for node in nodes.tolist():
        N.InsertNextPoint(node)
    E = vtkCellArray()
    for f in polys.tolist():
        E.InsertNextCell(len(f),f)
    N.Modified()
    E.Modified()
    polyd = vtkPolyData()
    polyd.SetPoints(N)
    polyd.SetPolys(E)

    if lines == True:
        edg = vtkExtractEdges()
        edg.UseAllPointsOn()
        edg.SetInputData(polyd)
        edg.Update()
        L = edg.GetOutput().GetLines()

    else:
        L = vtkCellArray()
        if isinstance(lines, np.ndarray):
            for l in lines.tolist():
                L.InsertNextCell(len(l), l)

    polyd.SetLines(L)

    V = vtkCellArray()
    if verts == True:
        verts = np.arange(nodes.shape[0])

    if isinstance(verts, np.ndarray):
        if verts.ndim == 1:
            verts = verts[...,None]
        for v in verts.tolist():
            V.InsertNextCell(len(v), v)

    polyd.SetVerts(V)

    return polyd

def polydata_from_plane(plane_abcd, bounds, polyd=None):
    plane_abcd = np.asarray(plane_abcd)
    # an implicit plane `plane_abcd` limited by `bounds`
    intersections = np.empty((18,), dtype=float)
    intersections[...] = float('nan')
    vtkBox.IntersectWithPlane(bounds, -plane_abcd[:-1]*plane_abcd[-1], plane_abcd[:-1], intersections)
    verts = intersections.reshape(-1,3)
    verts = verts[np.any(verts!=0, axis=1) & np.any(~np.isnan(verts), axis=1)]

    if polyd is None:
        polyd = vtkPolyData()

    points = vtkPoints()
    for p in verts:
        points.InsertNextPoint(p)
    n = points.GetNumberOfPoints()
    cells = vtkCellArray()
    cells.InsertNextCell(n, list(range(n)))
    # l = vtkPolygon()
    # l.GetPointIds().SetNumberOfIds(n)
    # for i in range(n):
    #     l.GetPointIds().SetId(i,i)
    # cells.InsertNextCell(l)
    polyd.SetPoints(points)
    polyd.SetPolys(cells)
    polyd.Modified()

    return polyd


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


def clip_polydata_with_mesh(image_polydata, mesh_polydata):
    # this function requires further refinement

    dist = vtkImplicitPolyDataDistance()
    dist.SetInput(mesh_polydata)

    clipper = vtkClipPolyData()
    clipper.SetClipFunction(dist)
    clipper.SetInputData(image_polydata)
    clipper.InsideOutOn()
    clipper.SetValue(0.0)
    clipper.GenerateClippedOutputOff()

    largest_region = vtkConnectivityFilter()
    largest_region.SetInputConnection(clipper.GetOutputPort())
    largest_region.SetExtractionModeToLargestRegion()
    largest_region.Update()
    clipped = largest_region.GetOutput()

    return clipped


def icp(source, target, init_pose=None, max_iterations=2000, tolerance=0.001):
    from simpleicp import PointCloud, SimpleICP
    import numpy as np

    # Create point cloud objects
    target = PointCloud(target, columns=["x", "y", "z"])
    source = PointCloud(source, columns=["x", "y", "z"])

    # Create simpleICP object, add point clouds, and run algorithm!
    icp = SimpleICP()
    icp.add_point_clouds(target, source)
    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(correspondences=min(target._num_points//2,source._num_points//2),min_change=.5)
    return H
    




# def rigid_register(source:vtkPolyData, target:vtkPolyData):

#     from pycpd import RigidRegistration
#     from vtk import vtkDecimatePro 
#     from vtkmodules.util.numpy_support import vtk_to_numpy

#     source = vtk_to_numpy(source.GetPoints().GetData())
#     target = vtk_to_numpy(target.GetPoints().GetData())

#     # create a RigidRegistration object
#     reg = RigidRegistration(X=target, Y=source)
#     # run the registration & collect the results
#     TY, (s_reg, R_reg, t_reg) = reg.register()   
#     T = np.eye(4)
#     T[:-1,:-1] = R_reg
#     T[:-1,-1] = t_reg
#     return T 




# def icp_register(source, target, return_numpy_instead=True):

#     T = vtkIterativeClosestPointTransform()
#     T.SetSource(source)
#     T.SetTarget(target)
#     T.Update()
#     if return_numpy_instead:
#         X = np.array([T.GetMatrix().GetElement(*ij) for ij in np.ndindex((4,4))]).reshape(4,4)
#     else:
#         X = T.GetMatrix()

#     return X


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
    for id in ind_2d:
        lines.InsertNextCell(len(id), id)

        # l = vtkPolyLine()
        # l.GetPointIds().SetNumberOfIds(id.size)

        # for i,k in enumerate(id):
        #     l.GetPointIds().SetId(i,k)

        # lines.InsertNextCell(l)

    return lines


def test():
    pass


if __name__=='__main__':
    test()


