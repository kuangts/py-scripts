import os, sys, glob
from scipy.interpolate import RBFInterpolator
from vtkmodules.vtkRenderingCore import vtkInteractorStyle

from .rendering import *
from .mesh import *
from .polydata import *

class Selector(Window):

    DEFAULT_STYLE_CLASS = vtkInteractorStyleTrackballCamera
    DEFAULT_PICKER_CLASS = vtkCellPicker
    DEFAULT_PICKER_TOLERANCE = 1e-6
    DEFAULT_POINT_SIZE = 10
    DEFAULT_COLOR_SERIES_IDX = 16 # use any integer realy
    DEFAULT_COLOR_LOWER_BOUND = -1.0
    DEFAULT_COLOR_UPPER_BOUND = +1.0


    @property
    def colormap(self):
        if hasattr(self, '_colormap'):
            return self._colormap
        else:
            return None

    @colormap.setter
    def colormap(self, cmap):
        if self.colormap:
            return ValueError('colormap exists, use set_color instead')
        self._colormap = cmap

    @property
    def COLOR_LOWER_BOUND(self):
        if self.colormap is None:
            return self.DEFAULT_COLOR_LOWER_BOUND
        else:
            return self.colormap.GetRange()[0]

    @property
    def COLOR_UPPER_BOUND(self):
        if self.colormap is None:
            return self.DEFAULT_COLOR_UPPER_BOUND
        else:
            return self.colormap.GetRange()[1]


    def __init__(self, **kwargs):
        super().__init__()
        self.attach_picker(**{k:v for k,v in kwargs.items() if k.startswith('picker')})
        return None


    def set_color(self, range=None, series_idx=None):
        if range is None:
            range = (self.DEFAULT_COLOR_LOWER_BOUND, self.DEFAULT_COLOR_UPPER_BOUND)
        if series_idx is None:
            series_idx = self.DEFAULT_COLOR_SERIES_IDX
        
        colormap = build_color_series(*range, series_idx, 
                                      lut=self.colormap)

        if not self.colormap:
            self.colormap = colormap

        return None


    def attach_picker(self, picker=None, picker_tolerance=None):
        if picker is None:
            picker = self.DEFAULT_PICKER_CLASS()
        self.picker = picker
        self.picker.SetTolerance(self.DEFAULT_PICKER_TOLERANCE if picker_tolerance is None else picker_tolerance)
        self.picker.InitializePickList()
        self.picker.SetPickFromList(True)
        

    def initialize_selection_indices(self):
        pass


    def add_pick_polydata(self, *polyds, **kwargs):
        for pld in polyds:
            prop = polydata_actor(pld, RenderPointsAsSpheres=True, PointSize=self.DEFAULT_POINT_SIZE, **kwargs)
            # register this prop in pick list and display
            self.renderer.AddActor(prop)
            self.picker.AddPickList(prop)
            # for subclasses
        
        self.initialize_selection_indices()

        return None


    def get_pick_props(self):
        l = self.picker.GetPickList()
        props = [vtkProp.SafeDownCast(l.GetItemAsObject(i)) for i in range(l.GetNumberOfItems())]
        return props


    def get_pick_mappers(self):
        return [vtkPolyDataMapper.SafeDownCast(p.GetMapper()) for p in self.get_pick_props()]


    def get_pick_properties(self):
        return [vtkActor.SafeDownCast(p).GetProperty() for p in self.get_pick_props()]


    def get_pick_polydata(self):
        return [m.GetInput() for m in self.get_pick_mappers()]


    def get_selection_indices(self):
        pass


class PolygonalSurfaceNodeSelector(Selector):

    def ctrl_left_action(self, obj:vtkInteractorStyle, event):

        if 'Press' in event or 'Move' in event:
            pos = obj.GetInteractor().GetEventPosition()
            self.picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
            id = self.picker.GetPointId()
            if id >= 0:
                verts = self.picker.GetDataSet().GetVerts()
                ids = vtk_to_numpy_(verts.GetConnectivityArray()).flatten()
                if id not in ids:
                    verts.InsertNextCell(1,[id])
                    verts.Modified()
                    self.render_window.Render()
                    print(f'number of ids selected: {ids.size}', end='\r')
            return None
        
        else:
            super().ctrl_left_action(obj, event)



class PolygonalSurfaceGlyphSelector(Selector):

    DEFAULT_GLYPH_COLOR = (1.0, 0.5, 0.5)
    DEFAULT_SELECTED_GLYPH_COLOR = (0.5, 1.0, 0.5)
    DEFAULT_CHANGED_GLYPH_COLOR = (1.0, 1.0, 0.5)


    def __init__(self, glyph_filter:vtkGlyph3D):
        super().__init__()

        glyph_filter.GeneratePointIdsOn()
        glyph_filter.Update()

        n_points = glyph_filter.GetInput().GetPoints().GetNumberOfPoints()
        self.colormap = vtkColorTransferFunction()
        for i in range(n_points):
            self.colormap.AddRGBPoint(i, *self.DEFAULT_GLYPH_COLOR)

        self.add_pick_polydata(glyph_filter.GetOutput())
        self.glyph_filter = glyph_filter
        self.glyph_picker = vtkCellPicker()
        self.glyph_picker.SetTolerance(self.DEFAULT_PICKER_TOLERANCE)
        self.glyph_picker.InitializePickList()
        self.glyph_picker.SetPickFromList(True)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph_filter.GetOutputPort())
        mapper.SetLookupTable(self.colormap)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray('InputPointIds')

        prop = vtkActor()
        prop.SetMapper(mapper)
        
        self.renderer.AddActor(prop)
        self.glyph_picker.AddPickList(prop)
        self.glyph_id = None

        return None


    def left_button_press_event(self, obj: vtkInteractorStyle, event):

        pos = obj.GetInteractor().GetEventPosition()
        self.glyph_picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
        
        if self.glyph_picker.GetCellId() >= 0:

            # direct hit on to one of the landmarks
            glyph_polyd = self.glyph_filter.GetOutput()
            if glyph_polyd == self.glyph_picker.GetDataSet():

                # just made sure it is indeed hit on a landmark glyph
                # figure out which landmark it is
                ids = glyph_polyd.GetPointData().GetArray('InputPointIds')
                id_ = vtkIdTypeArray.SafeDownCast(ids).GetTuple(self.glyph_picker.GetPointId())[0]
                id_ = int(id_)

                if self.glyph_id != id_:

                    # selected a new landmark, so no further action
                    self.update_glyph(ind=id_)
                    print(f'selected {self.glyph_id}')

                    return None

        return super().left_button_press_event(obj, event)


    def ctrl_left_action(self, obj: vtkInteractorStyle, event):

        if 'Press' in event:

            if self.glyph_id is not None:

                # a landmark is currently selected, so update its location
                pos = obj.GetInteractor().GetEventPosition()
                self.picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())

                if self.picker.GetCellId() >= 0:
                    self.update_glyph(coords=self.picker.GetPickPosition())
                print(f'updated {self.glyph_id}, to {self.picker.GetPickPosition()}')
            
            else:

                # misses the landmarks
                # deselect the current landmark, if any
                self.glyph_id == None

            return None
        

    def update_glyph(self, ind=None, coords=None):

        if ind is not None:
            if self.glyph_id is not None:
                self.colormap.RemovePoint(self.glyph_id)
                self.colormap.AddRGBPoint(self.glyph_id, *self.DEFAULT_CHANGED_GLYPH_COLOR)
            self.colormap.AddRGBPoint(ind, *self.DEFAULT_SELECTED_GLYPH_COLOR)
            self.glyph_id = ind

        else:
            ind = self.glyph_id

        if coords:
            self.glyph_filter.GetInput().GetPoints().SetPoint(ind, coords)
            self.glyph_filter.GetInput().GetPoints().Modified()

        self.render_window.Render()
        return None


# class CellSelector(Selector): # needs work

#     def __init__(self):
#         super().__init__('cell')


#     def initialize_selection_indices(self, *polyds):
#         for polyd in polyds:
#             pd = polyd.GetCellData()
#             if pd.HasArray('selection'):
#                 continue
#             ar = vtkCharArray()
#             ar.SetName('selection')
#             ar.SetNumberOfComponents(1)
#             ar.SetNumberOfTuples(polyd.GetNumberOfPolys())
#             pd.SetScalars(ar)
#             vtk_to_numpy_(ar)[...] = 0
#         return None


#     def get_selection_indices(self, polyd:vtkPolyData):
#         sel = polyd.GetCellData().GetScalars('selection')
#         # print(numpy.sum(vtk_to_numpy_(sel)!=0))
#         return sel


#     def select(self):
#         id = self.picker.GetCellId()
#         if id != -1:
#             coord = self.picker.GetPickPosition()
#             polyd = vtkPolyData.SafeDownCast(self.picker.GetDataSet())
#             sel = vtkCharArray.SafeDownCast(polyd.GetCellData().GetScalars('selection'))
#             sel.SetTuple(id, [1])
#             sel.Modified()
#             self.get_selection_indices(polyd)


class HexMeshSurfaceSelector(PolygonalSurfaceNodeSelector):

    CURVATURE_LOWER_BOUND = -.3
    CURVATURE_UPPER_BOUND = +.3
    DEFAULT_COLOR_LOWER_BOUND = CURVATURE_LOWER_BOUND
    DEFAULT_COLOR_UPPER_BOUND = CURVATURE_UPPER_BOUND
    OTHER_COLOR = 'Gray'

    def __init__(self, inp_file):

        super().__init__()
        self.set_color((self.CURVATURE_LOWER_BOUND, self.CURVATURE_UPPER_BOUND), self.DEFAULT_COLOR_SERIES_IDX)

        nodes, elems = read_inp(inp_file)
        nodes = nodes.astype(np.float64)
        elems = elems.astype(np.int64)
        faces = boundary_faces(elems, dims=((False, False), (False, True), (False, False)))
        other_faces = boundary_faces(elems, dims=((True, True), (True, False), (True, True)))
        node_grid = calculate_grid(nodes, elems)

        self.nodes = vtkPoints()
        self.elems = vtkCellArray()
        self.node_grid = vtkIdTypeArray()
        self.nodes.SetData(
            numpy_to_vtk(num_array=nodes, deep=True, array_type=VTK_DOUBLE)
            )
        self.elems.SetData(
            numpy_to_vtkIdTypeArray(numpy.arange(0, elems.size+1, elems.shape[1], dtype=VTK_ID_DTYPE), deep=True), 
            numpy_to_vtkIdTypeArray(elems.ravel(), deep=True)
            )
        self.node_grid = numpy_to_vtkIdTypeArray(node_grid, deep=True)

        self.faces = vtkCellArray()
        self.faces.SetData(
            numpy_to_vtkIdTypeArray(numpy.arange(0, faces.size+1, faces.shape[1], dtype=VTK_ID_DTYPE), deep=True), 
            numpy_to_vtkIdTypeArray(faces.ravel(), deep=True)
            )

        self.other_faces = vtkCellArray()
        self.other_faces.SetData(
            numpy_to_vtkIdTypeArray(numpy.arange(0, other_faces.size+1, other_faces.shape[1], dtype=VTK_ID_DTYPE), deep=True), 
            numpy_to_vtkIdTypeArray(other_faces.ravel(), deep=True)
            )

        self.polyd = vtkPolyData()
        self.polyd.SetPoints(self.nodes)
        self.polyd.SetPolys(self.faces)
        add_edges(self.polyd)
        self.other_polyd = vtkPolyData()
        self.other_polyd.SetPoints(self.nodes)
        self.other_polyd.SetPolys(self.other_faces)

        self.node_grid.SetName('NODE_GRID')
        self.polyd.GetPointData().SetVectors(self.node_grid)
        self.polyd.GetPointData().SetActiveVectors('NODE_GRID')


        cleaner = vtkCleanPolyData()
        cleaner.SetInputData(self.polyd)
        curver = vtkCurvatures()
        curver.SetCurvatureTypeToMean()
        curver.SetInputConnection(cleaner.GetOutputPort())
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(curver.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray('Mean_Curvature')
        mapper.SetScalarRange(self.CURVATURE_LOWER_BOUND, self.CURVATURE_UPPER_BOUND)
        mapper.SetLookupTable(self.colormap)
        mapper.Update()
        actor = vtkActor()
        actor.SetMapper(mapper)
        self.polyd = curver.GetOutput()
        self.curv_filter = curver
        grid = curver.GetOutput().GetPointData().GetArray('NODE_GRID')
        curv = curver.GetOutput().GetPointData().GetArray('Mean_Curvature')
        print(curv.GetRange())
        self.renderer.AddActor(actor)
        self.picker.AddPickList(actor)
        # curv = adjust_edge_curvatures(cc.GetOutput(), curvature_name)
        
        other_prop = polydata_actor(self.other_polyd, Color=self.OTHER_COLOR, Opacity=.5)
        other_prop.SetObjectName('OTHER_PROP')
        self.renderer.AddActor(other_prop)

        return None


    def space_action(self):
        

        nodes = vtkpoints_to_numpy_(self.nodes)
        elems = vtkpolys_to_numpy_(self.elems)
        node_grid = vtk_to_numpy_(self.node_grid)
        poly_grid = vtk_to_numpy_(self.curv_filter.GetOutput().GetPointData().GetArray('NODE_GRID'))
        ind_node_select = vtkpolys_to_numpy_(self.curv_filter.GetOutput().GetVerts()).flatten()

        ind_node_change_3d = np.all(node_grid[:,None,:] == poly_grid[ind_node_select,:][None,:,:], axis=2)
        d1,d2 = np.nonzero(ind_node_change_3d)
        ind_node_change = d1[np.argsort(d2)]

        ind_node_faces = np.nonzero(np.logical_or(node_grid[:,1]==node_grid[:,1].min(), node_grid[:,1]==node_grid[:,1].max()))[0]
        ind_node_change_neighbor = ind_node_change.copy()
        ind_node_change_neighbor = np.all(node_grid[:,None,(0,2)] == node_grid[ind_node_change,:][None,:,(0,2)], axis=2)
        ind_node_change_neighbor = np.nonzero(ind_node_change_neighbor.any(axis=1))[0]
        for _ in range(2):
            ind_node_change_neighbor = elems[np.isin(elems,ind_node_change_neighbor).any(axis=1),:].flatten()

        ind_node_change_neighbor = np.unique(ind_node_change_neighbor)
        ind_node_change_neighbor_interp = ind_node_change_neighbor.copy()
        for _ in range(3):
            ind_node_change_neighbor_interp = elems[np.isin(elems,ind_node_change_neighbor_interp).any(axis=1),:].flatten()

        ind_node_change_neighbor_interp = np.unique(ind_node_change_neighbor_interp)
        ind_node_change = np.union1d(ind_node_change, np.setdiff1d(ind_node_change_neighbor, ind_node_faces))
        ind_node_interp = np.setdiff1d(ind_node_change_neighbor_interp, ind_node_change)

        X = nodes[ind_node_change,:]
        nodes[ind_node_change,:] = RBFInterpolator(node_grid[ind_node_interp,:], nodes[ind_node_interp,:], degree=3)(node_grid[ind_node_change,:])
        print(f'maximum movement: {np.max(np.sum((nodes[ind_node_change,:]-X)**2,axis=1)**.5)}')
        self.nodes.Modified()
        self.polyd.GetVerts().Reset()
        self.curv_filter.Update()
        self.render_window.Render()

        return None


