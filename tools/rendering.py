import sys
from typing import Union
import numpy
from vtkmodules.vtkCommonColor import vtkNamedColors, vtkColorSeries
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkImageData
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkMapper,
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
    vtkColorTransferFunction,
    
)
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkExtractEdges, vtkIdFilter
from vtkmodules.vtkFiltersGeneral import vtkCurvatures
from vtkmodules.vtkCommonCore import vtkDataArray, vtkScalarsToColors, vtkCharArray, vtkLookupTable
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import vtkProp, vtkInteractorStyle, vtkBillboardTextActor3D
from vtk_bridge import *
from .polydata import *
from register import nicp # cannot run without this, don't know why
from vtkmodules.util import numpy_support
from vtkmodules.numpy_interface import dataset_adapter as dsa


colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']

colors = vtkNamedColors()
color_series = vtkColorSeries()

# https://htmlpreview.github.io/?https://github.com/Kitware/vtk-examples/blob/gh-pages/VTKColorSeriesPatches.html


def build_colors(lower_bound, upper_bound, color_scheme_idx:int=15, colors_rgb:np.ndarray=None):

    if color_scheme_idx is not None:
        color_series = vtkColorSeries()
        color_series.SetColorScheme(color_scheme_idx)
        lookup_table = color_series.CreateLookupTable()

    elif colors_rgb is not None:
        lookup_table = vtkLookupTable()
        for i, (r,g,b) in enumerate(colors_rgb):
            lookup_table.SetTableValue(i,r,g,b)

    lookup_table.SetTableRange(lower_bound, upper_bound)
    lookup_table.Modified()
    lookup_table.Build()

    return lookup_table


def update_colors(lookup_table:vtkLookupTable, color_scheme_idx:int=None, colors_rgb:np.ndarray=None, lower_bound:float=None, upper_bound:float=None):

    if color_scheme_idx is not None:
        lookup_table.ResetAnnotations()
        color_series = vtkColorSeries()
        color_series.SetColorScheme(color_scheme_idx)
        color_series.BuildLookupTable(lookup_table)

    elif colors_rgb is not None:
        lookup_table.ResetAnnotations()
        for i, (r,g,b) in enumerate(colors_rgb):
            lookup_table.SetTableValue(i,r,g,b)
    
    if lower_bound is None:
        lower_bound = lookup_table.GetTableRange()[0]
    
    if upper_bound is None:
        upper_bound = lookup_table.GetTableRange()[1]

    lookup_table.SetTableRange(lower_bound, upper_bound)
    lookup_table.Modified()
    lookup_table.Build()

    return None


def map_scalars_through_table(lut:vtkScalarsToColors, scalars:Union[numpy.ndarray, vtkDataArray]) -> numpy.ndarray:
    if isinstance(scalars, numpy.ndarray):
        sc = scalars.flatten()
        rgba = numpy.empty(sc.size*4, "uint8")
        sc_vtk = numpy_to_vtk_(sc)
        shp = scalars.shape
    else:
        sc_vtk = scalars
        shp = (scalars.GetNumberOfTuples(),)
    rgba = numpy.empty(sc_vtk.GetNumberOfTuples()*4, "uint8")
    lut.MapScalarsThroughTable(sc_vtk, rgba)
    rgba = rgba.reshape(*shp,4)/255 #???????
    return rgba


# def render_window(window_title=''):
#     renderer = vtkRenderer()
#     renderer.SetBackground(.67, .93, .93)

#     render_window = vtkRenderWindow()
#     render_window.AddRenderer(renderer)
#     render_window.SetSize(1000,1500)
#     render_window.SetWindowName(window_title)

#     interactor = vtkRenderWindowInteractor()
#     interactor.SetRenderWindow(render_window)

#     style = vtkInteractorStyleTrackballCamera()
#     style.SetDefaultRenderer(renderer)
#     interactor.SetInteractorStyle(style)
#     return interactor, renderer


def set_curvatures(polyd, curvature_name):

    def adjust_edge_curvatures(source, curvature_name, epsilon=1.0e-08):

        """
        This function adjusts curvatures along the edges of the surface by replacing
        the value with the average value of the curvatures of points in the neighborhood.

        Remember to update the vtkCurvatures object before calling this.

        :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
        :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
        :param epsilon: Absolute curvature values less than this will be set to zero.
        :return:
        """
        # https://examples.vtk.org/site/Python/PolyData/Curvatures/

        def point_neighbourhood(pt_id):
            """
            Find the ids of the neighbours of pt_id.

            :param pt_id: The point id.
            :return: The neighbour ids.
            """
            """
            Extract the topological neighbors for point pId. In two steps:
            1) source.GetPointCells(pt_id, cell_ids)
            2) source.GetCellPoints(cell_id, cell_point_ids) for all cell_id in cell_ids
            """
            cell_ids = vtkIdList()
            source.GetPointCells(pt_id, cell_ids)
            neighbour = set()
            for cell_idx in range(0, cell_ids.GetNumberOfIds()):
                cell_id = cell_ids.GetId(cell_idx)
                cell_point_ids = vtkIdList()
                source.GetCellPoints(cell_id, cell_point_ids)
                for cell_pt_idx in range(0, cell_point_ids.GetNumberOfIds()):
                    neighbour.add(cell_point_ids.GetId(cell_pt_idx))
            return neighbour

        def compute_distance(pt_id_a, pt_id_b):
            """
            Compute the distance between two points given their ids.

            :param pt_id_a:
            :param pt_id_b:
            :return:
            """
            pt_a = np.array(source.GetPoint(pt_id_a))
            pt_b = np.array(source.GetPoint(pt_id_b))
            return np.linalg.norm(pt_a - pt_b)

        # Get the active scalars
        source.GetPointData().SetActiveScalars(curvature_name)
        curvatures = vtk_to_numpy_(source.GetPointData().GetScalars(curvature_name)).flatten().copy()

        #  Get the boundary point IDs.
        array_name = 'ids'
        id_filter = vtkIdFilter()
        id_filter.SetInputData(source)
        id_filter.SetPointIds(True)
        id_filter.SetCellIds(False)
        id_filter.SetPointIdsArrayName(array_name)
        id_filter.SetCellIdsArrayName(array_name)
        id_filter.Update()

        edges = vtkFeatureEdges()
        edges.SetInputConnection(id_filter.GetOutputPort())
        edges.BoundaryEdgesOn()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.FeatureEdgesOff()
        edges.Update()

        edge_array = edges.GetOutput().GetPointData().GetArray(array_name)
        boundary_ids = []
        for i in range(edges.GetOutput().GetNumberOfPoints()):
            boundary_ids.append(edge_array.GetValue(i))
        # Remove duplicate Ids.
        p_ids_set = set(boundary_ids)

        # Iterate over the edge points and compute the curvature as the weighted
        # average of the neighbours.
        count_invalid = 0
        for p_id in boundary_ids:
            p_ids_neighbors = point_neighbourhood(p_id)
            # Keep only interior points.
            p_ids_neighbors -= p_ids_set
            # Compute distances and extract curvature values.
            curvs = [curvatures[p_id_n] for p_id_n in p_ids_neighbors]
            dists = [compute_distance(p_id_n, p_id) for p_id_n in p_ids_neighbors]
            curvs = np.array(curvs)
            dists = np.array(dists)
            curvs = curvs[dists > 0]
            dists = dists[dists > 0]
            if len(curvs) > 0:
                weights = 1 / np.array(dists)
                weights /= weights.sum()
                new_curv = np.dot(curvs, weights)
            else:
                # Corner case.
                count_invalid += 1
                # Assuming the curvature of the point is planar.
                new_curv = 0.0
            # Set the new curvature value.
            curvatures[p_id] = new_curv

        
        #  Set small values to zero.
        if epsilon != 0.0:
            curvatures = np.where(abs(curvatures) < epsilon, 0, curvatures)

        curv = numpy_to_vtk(num_array=curvatures.ravel(), deep=True, array_type=VTK_DOUBLE)
        curv.SetName(curvature_name)

        return curv
    
    cc = vtkCurvatures()
    cc.SetInputData(polyd)
    if curvature_name == 'Mean_Curvature':
        cc.SetCurvatureTypeToMean()
    elif curvature_name == 'Gauss_Curvature':
        cc.SetCurvatureTypeToGaussian()
    else:
        raise ValueError('add support later')
    cc.Update()
    curv = cc.GetOutput().GetPointData().GetArray(curvature_name)
    # curv = adjust_edge_curvatures(cc.GetOutput(), curvature_name)

    if polyd.GetPointData().HasArray(curvature_name):
        polyd.GetPointData().RemoveArray(curvature_name)
    polyd.GetPointData().AddArray(curv)
    polyd.GetPointData().SetActiveScalars(curvature_name)
    return None


def add_edges(polyd):
    # add edges to polydata
    edg = vtkExtractEdges()
    edg.UseAllPointsOn()
    edg.SetInputData(polyd)
    edg.Update()
    polyd.SetLines(edg.GetOutput().GetLines())


def polydata_actor(polyd:vtkPolyData, mapper=None, **property):
    if mapper is None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polyd)
    actor = vtkActor()
    actor.SetMapper(mapper)
    if property:
        for pk,pv in property.items():
            if pk=='Color':
                if isinstance(pv, int):
                    pv = colornames[pv]
                if isinstance(pv, str):
                    pv = colors.GetColor3d(pv)
            getattr(actor.GetProperty(),'Set'+pk).__call__(pv)

    return actor


def text_actor(coords:numpy.ndarray, label:str, font_size=24, color=(0,0,0), display_offset=(0,10), **text_property):
    if isinstance(color, str):
        color = colors.GetColor3d(color)

    actor = vtkBillboardTextActor3D()
    actor.SetPosition(coords)
    actor.SetInput(label)
    actor.SetDisplayOffset(*display_offset)
    actor.GetTextProperty().SetFontSize(font_size)
    actor.GetTextProperty().SetColor(color)
    actor.GetTextProperty().SetJustificationToCentered()
    actor.PickableOff()
    # actor.ForceOpaqueOn()
    if text_property:
        for pk,pv in text_property.items():
            getattr(actor.GetTextProperty(),'Set'+pk).__call__(pv)

