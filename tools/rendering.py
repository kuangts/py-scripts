import sys
from typing import Union
import numpy
from vtkmodules.vtkCommonColor import vtkNamedColors, vtkColorSeries
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkImageData
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
    vtkColorTransferFunction,
    
)
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkExtractEdges, vtkIdFilter
from vtkmodules.vtkFiltersGeneral import vtkCurvatures
from vtkmodules.vtkCommonCore import vtkDataArray, vtkScalarsToColors, vtkCharArray
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

def build_color(lower_bound, upper_bound, symmetric_map=False) -> vtkColorTransferFunction:
    lut = vtkColorTransferFunction()
    r = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5]
    g = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    b = [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rgb = numpy.vstack((r,g,b)).T.astype(numpy.float64)
    if symmetric_map:
        rgb = numpy.vstack((rgb[:0:-1,:], rgb))
    lut.BuildFunctionFromTable(lower_bound, upper_bound, rgb.shape[0], rgb.ravel())
    lut.Modified()
    return lut


def build_color_series(lower_bound:float, upper_bound:float, color_map_idx=16, lut:vtkScalarsToColors=None):
    color_series = vtkColorSeries()
    color_series.SetColorScheme(color_map_idx)
    # print(f'Using color scheme #: {color_series.GetColorScheme()}, {color_series.GetColorSchemeName()}')

    if lut is None:
        lut = vtkColorTransferFunction()
        lut.SetColorSpaceToHSV()
    else:
        lut.RemoveAllPoints()

    # Use a color series to create a transfer function
    for i in range(0, color_series.GetNumberOfColors()):
        color = color_series.GetColor(i)
        double_color = list(map(lambda x: x / 255.0, color))
        t = lower_bound + (upper_bound - lower_bound) / (color_series.GetNumberOfColors() - 1) * i
        lut.AddRGBPoint(t, double_color[0], double_color[1], double_color[2])

    return lut


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


def render_window(window_title=''):
    renderer = vtkRenderer()
    renderer.SetBackground(.67, .93, .93)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000,1500)
    render_window.SetWindowName(window_title)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)
    return interactor, renderer


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
    
    polyd_in = vtkPolyData()
    polyd_in.SetPoints(polyd.GetPoints())
    polyd_in.SetPolys(polyd.GetPolys())
    cc = vtkCurvatures()
    cc.SetInputData(polyd_in)
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
    polyd.GetPointData().SetScalars(curv)
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


class Window():

    DEFAULT_STYLE_CLASS = vtkInteractorStyleTrackballCamera

    def __init__(self):

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(.67, .93, .93)

        self.render_window = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000,1500)
        self.render_window.SetWindowName('')

        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.attach_style()
        return None


    def attach_style(self, style=None):
        if style is None:
            self.style = self.DEFAULT_STYLE_CLASS()
            # Interactor callbacks
            self.style.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
            self.style.AddObserver('LeftButtonReleaseEvent', self.left_button_release_event)
            self.style.AddObserver('RightButtonPressEvent', self.right_button_press_event)
            self.style.AddObserver('RightButtonReleaseEvent', self.right_button_release_event)
            self.style.AddObserver('MouseMoveEvent', self.mouse_move_event)
            self.style.AddObserver('KeyPressEvent', self.key_press_event)
        else:
            self.style = style

        self.button_status = dict(left=0, right=0)
        self.style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(self.style)
        return None


    def start(self):
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()
        return None


    def quit(self):
        self.render_window.Finalize()
        self.interactor.TerminateApp()
        del self.render_window, self.interactor


    def refresh(self, text=None):
        self.render_window.Render()


    def resolve_key(self, key):
        # print(key)
        if key=='space':
            print('\n', end='')
            self.space_action()
        self.render_window.Render()


    def space_action(self):
        pass


    def ctrl_left_action(self, obj:vtkInteractorStyle, event):
        # this is to be overridden by subclass
        # should be unique for each class
        pass
        

    # Interactor callbacks
    def key_press_event(self, obj:vtkInteractorStyle, event):
        return self.resolve_key(obj.GetInteractor().GetKeySym())

    def left_button_press_event(self, obj:vtkInteractorStyle, event):
        self.button_status['left'] = 1
        if obj.GetInteractor().GetControlKey():
            return self.ctrl_left_action(obj, event)
        else:
            return obj.OnLeftButtonDown()

    def left_button_release_event(self, obj:vtkInteractorStyle, event):
        self.button_status['left'] = 0
        if obj.GetInteractor().GetControlKey():
            return self.ctrl_left_action(obj, event)
        else:
            return obj.OnLeftButtonUp()

    def right_button_press_event(self, obj:vtkInteractorStyle, event):
        self.button_status['right'] = 1
        return obj.OnRightButtonDown()

    def right_button_release_event(self, obj:vtkInteractorStyle, event):
        self.button_status['right'] = 0
        return obj.OnRightButtonUp()

    def mouse_move_event(self, obj:vtkInteractorStyle, event):
        if self.button_status['left'] and obj.GetInteractor().GetControlKey():
            return self.ctrl_left_action(obj, event)
        else:
            return obj.OnMouseMove()


