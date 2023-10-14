from enum import Enum
from tools.ui import *
from vtk import *


class Key(Enum):
    Ctrl=0
    Alt=1
    Shift=2


class Button(Enum):
    Left=0
    Right=1


class vtkLandmark(vtkTable):

    allow_select = True
    allow_edit = True
    allow_add = True
    select_key = Key.Alt
    edit_key = Key.Ctrl
    add_key = Key.Shift
    picker_tolerance = 1e-4
    

    def __init__(self) -> None:
        name = vtkStringArray()
        name.SetName('Name')
        coord = vtkFloatArray()
        coord.SetNumberOfComponents(3)
        coord.SetName('Coordinates')
        self.AddColumn(name)
        self.AddColumn(coord)

        super().__init__()


    @property
    def size(self):
        return self.GetNumberOfRows()
    

    @property
    def names(self):
        return self.GetColumnByName('Name')
    
    def get_name(self, i):
        return self.names.GetValue(i)
    
    @property
    def points(self):
        return self.GetColumnByName('Coordinates')
    

    def get_point(self, i):
        return self.points.GetTuple(i)
    

    def add_point(self, name, coordinates):
        rid = self.InsertNextBlankRow()
        self.GetColumnByName('Name').SetValue(rid, name)
        self.set_point(rid, coordinates)


    def set_point(self, rid, coordinates):
        self.GetColumnByName('Coordinates').SetTuple(rid, coordinates)


    def remove_point(self, i):
        self.RemoveRow(i)
        self.SqueezeRows()
        return None
    

    @property
    def pointset(self):
        if not hasattr(self, '_pointset'):
            pset = vtkPointSet()
            pnts = vtkPoints()
            pset.SetPoints(pnts)
            pnts.SetData(self.points)
            self._pointset = pset

        return self._pointset
    
    def glyph(self, sphere_radius=.5):
        
        if not hasattr(self, '_glyph'):

            glyphSource = vtkSphereSource()
            glyphSource.SetRadius(sphere_radius)
            glyphSource.Update()

            glyph3D = vtkGlyph3D()
            glyph3D.GeneratePointIdsOn()
            glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
            glyph3D.SetInputData(self.pointset)
            glyph3D.SetScaleModeToDataScalingOff()
            glyph3D.Update()

            self._glyph = glyph3D

        return self._glyph



    def set_allow_select(self, select:bool):
        self.allow_select = select
        if select:
            self.picker.AddPickList(self._glyph_actor)
        else:
            self.picker.DeletePickList(self._glyph_actor)


    def set_allow_edit(self, edit:bool):
        self.allow_edit = edit
        self.set_allow_select(True)


    def set_allow_add(self, add:bool):
        self.allow_add = add


    def init(self, **kwargs):
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph().GetOutputPort())
        self._glyph_actor = vtkActor()
        self._glyph_actor.SetMapper(mapper)
        
        
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(self.picker_tolerance)
        self.picker.InitializePickList()
        self.picker.SetPickFromList(True)

        self.current_selection = None
        if self.allow_select or self.allow_edit:
            self.picker.AddPickList(self._glyph_actor)


    def _next_name(self):
        return 'asdf'


    def pick(self, position, renderer, func_keys=[]):

        self.picker.Pick(position[0], position[1], 0, renderer)
        if self.picker.GetCellId() >= 0:
            picked_actor = self.picker.GetActor()
        else:
            picked_actor = None
        handled = False

        if len(func_keys)>1 :
            print('this class does not accept multiple function keys')
            return handled
        
        if self.select_key in func_keys and self.allow_select:
            handled = True
            if picked_actor == self._glyph_actor:
                ids = self.picker.GetDataSet().GetPointData().GetArray('InputPointIds')
                id = vtkIdTypeArray.SafeDownCast(ids).GetTuple(self.picker.GetPointId())[0]
                self.current_selection = int(id)
                print(f'selected {self.current_selection}')
            else:
                self.current_selection = None
                print(f'selected {self.current_selection}')

        if self.edit_key in func_keys and self.allow_edit and self.current_selection is not None and picked_actor != self._glyph_actor:
            handled = True
            self.set_point(self.current_selection, self.picker.GetPickPosition())
            print(f'moved {self.current_selection} to {self.get_point(self.current_selection)}')
            self.points.Modified()


        if self.add_key in func_keys and self.allow_add and self.current_selection is None and picked_actor != self._glyph_actor:
            handled = True
            self.add_point(self._next_name(), self.picker.GetPickPosition())
            print(f'added {self.size-1} at {self.picker.GetPickPosition()}')
            self.points.Modified()

        return handled


    # def _delete(self):
    #     if not self.mode.allows_delete() or self.current_id is None:
    #         return None
        
    #     name = self.selection_names[self.current_id]
    #     self.selection_points.GetData().RemoveTuple(self.current_id)
    #     self.selection_points.Modified()
    #     del self.selection_names[self.current_id]
    #     self.lookup_table.AddRGBPoint(self.current_id, *self.DEFAULT_GLYPH_COLOR)
    #     self.lookup_table.RemovePoint(self.selection_points.GetNumberOfPoints()-1)
    #     self.current_id = None
    #     self.legend.SetInput(f'{name} deleted')
    #     self.render_window.Render()
    #     return None
    


class WindowWithSpecializedPicker(Window):


    def left_button_press_event(self):

        pos = self.style.GetInteractor().GetEventPosition()
        func_keys = []

        if self.style.GetInteractor().GetAltKey():
            func_keys.append(Key.Alt)
        if self.style.GetInteractor().GetControlKey():
            func_keys.append(Key.Ctrl)
        if self.style.GetInteractor().GetShiftKey():
            func_keys.append(Key.Shift)

        if self.lmk.pick(pos, self.style.GetDefaultRenderer(), func_keys):
            self.render_window.Render()
            return None
        else:
            return super().left_button_press_event()


lmk = vtkLandmark()
lmk.add_point('a', (*np.random.rand(3,)*100,))
lmk.add_point('b', (*np.random.rand(3,)*100,))
lmk.add_point('c', (*np.random.rand(3,)*100,))
lmk.add_point('d', (*np.random.rand(3,)*100,))
lmk.add_point('e', (*np.random.rand(3,)*100,))
lmk.init()


win = WindowWithSpecializedPicker()
win.lmk = lmk
win.renderer.AddActor(win.lmk._glyph_actor)
stl = polydata_from_stl(r'C:\py-scripts\hexmesh\template\closed.stl')
act = win.add_polydata(stl)
win.lmk.picker.AddPickList(act)
win.initialize()
win.start()



