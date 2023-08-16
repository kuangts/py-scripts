import sys
from enum import IntFlag
from enum import auto as enum_auto
import numpy as np
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
from vtkmodules.vtkFiltersGeneral import vtkCurvatures
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkIdTypeArray, vtkPoints
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkProp, vtkInteractorStyle, vtkGlyph3DMapper
from vtkmodules.vtkFiltersSources import vtkSphereSource 


# from .polydata import *


class MOUSE(IntFlag):
    LEFT = enum_auto()
    RIGHT = enum_auto()
    MOVED = enum_auto()
    CTRL = enum_auto()
    ALT = enum_auto()
    
    # def left_down(self):
    #     return (self & self.LEFT) != 0

    # def left_moved(self):
    #     return (self & (self.LEFT | self.MOVED)) != 0

    # def right_down(self):
    #     return (self & self.RIGHT) != 0

    # def right_moved(self):
    #     return (self & (self.RIGHT | self.MOVED)) != 0

    # def ctrl_left(self):
    #     return (self & (self.LEFT | self.CTRL)) != 0

    # def alt_left(self):
    #     return (self & (self.LEFT | self.ALT)) != 0

    # def ctrl_right(self):
    #     return (self & (self.RIGHT | self.CTRL)) != 0

    # def alt_right(self):
    #     return (self & (self.RIGHT | self.ALT)) != 0


class MODE(IntFlag):
    SELECT = enum_auto()
    _MOD = enum_auto()
    _DEL = enum_auto()
    ADD = enum_auto()
    MODIFY = _MOD | SELECT
    DELETE = _DEL | SELECT

    def allows_select(self):
        return (self & self.SELECT) != 0

    def allows_add(self):
        return (self & self.ADD) != 0

    def allows_modify(self):
        return (self & self.MODIFY) != 0

    def allows_delete(self):
        return (self & self.DELETE) != 0


class Window():

    DEFAULT_STYLE_CLASS = vtkInteractorStyleTrackballCamera
    
    @staticmethod
    def get_diverging_lut():
        """
        See: [Diverging Color Maps for Scientific Visualization](https://www.kennethmoreland.com/color-maps/)
                        start point         midPoint            end point
        cool to warm:     0.230, 0.299, 0.754 0.865, 0.865, 0.865 0.706, 0.016, 0.150
        purple to orange: 0.436, 0.308, 0.631 0.865, 0.865, 0.865 0.759, 0.334, 0.046
        green to purple:  0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.436, 0.308, 0.631
        blue to brown:    0.217, 0.525, 0.910 0.865, 0.865, 0.865 0.677, 0.492, 0.093
        green to red:     0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.758, 0.214, 0.233

        :return:
        """
        ctf = vtkColorTransferFunction()
        ctf.SetColorSpaceToDiverging()
        # Cool to warm.
        ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
        ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
        ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)

        table_size = 256
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(table_size)
        lut.Build()

        for i in range(0, table_size):
            rgba = list(ctf.GetColor(float(i) / table_size))
            rgba.append(1)
            lut.SetTableValue(i, rgba)

        return lut



    def __init__(self):

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(.67, .93, .93)

        self.render_window = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(960, 960)
        self.render_window.SetWindowName('')

        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.lut = self.get_diverging_lut()

        self.attach_style()
        return None


    def attach_style(self, style=None):
        if style is None:
            self.style = self.DEFAULT_STYLE_CLASS()
            # Interactor callbacks
            self.style.AddObserver('LeftButtonPressEvent', self.mouse_event)
            self.style.AddObserver('LeftButtonReleaseEvent', self.mouse_event)
            self.style.AddObserver('RightButtonPressEvent', self.mouse_event)
            self.style.AddObserver('RightButtonReleaseEvent', self.mouse_event)
            self.style.AddObserver('MouseMoveEvent', self.mouse_event)
            self.style.AddObserver('KeyPressEvent', self.key_event)
            self.style.AddObserver('KeyReleaseEvent', self.key_event)
        else:
            self.style = style

        self.mouse_status = 0
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


    def key_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()

        if event=='KeyPressEvent':
            # disable auto-repeat by monitoring held keys
            if not hasattr(self, '_keys'):
                self._keys = []
            if key in self._keys:
                return None
            else:
                self._keys.append(key)

            print('event:' + key + ' pressed')
            
            return self.key_press_event(key)
        elif event=='KeyReleaseEvent':
            if not hasattr(self, '_keys'):
                self._keys = []
            if key in self._keys:
                self._keys.remove(key)

            print('event:' + key + ' released')

            return self.key_release_event(key)


    def key_press_event(self, key):
        if key == 's' and self.interactor.GetControlKey():
            self.save()

        return None


    def save(self):
        pass


    def save_ui(self, **kwargs):

        from tkinter import Tk, filedialog
        Tk().withdraw()
    
        try:
            with filedialog.asksaveasfile(**kwargs) as f:
                self.save_file(f)
        except:
            print("file didn't save")        


    def save_file(self, f):
        pass


    def key_release_event(self, key):
        pass


    def mouse_event(self, obj:vtkInteractorStyle, event):
        # this is to be overridden by subclass
        # should be unique for each class

        # check if function keys are pressed
        # the default behavior is:
        # CONTINUE THE EVENT CYCLE AS PRESSED, EVEN IF FUNCTION KEY IS RELEASE LATER
        # if this is not desirable
        # monitor the `GetControlKey` instead

        print(f'event:{event}')
        
        if 'PressEvent' in event: 

            # both buttons cannot be pressed
            # one button press releases others
            if self.mouse_status & MOUSE.RIGHT:
                self.right_button_release_event()
            elif self.mouse_status & MOUSE.LEFT:
                self.left_button_release_event()
            # start of a fresh event cycle - reset mouse status
            self.mouse_status = 0

            # check ctrl key
            if obj.GetInteractor().GetControlKey():
                self.mouse_status |= MOUSE.CTRL
            # check alt key
            if obj.GetInteractor().GetAltKey():
                self.mouse_status |= MOUSE.ALT
            # if left button is pressed
            if event == 'LeftButtonPressEvent':
                self.mouse_status |= MOUSE.LEFT
                self.left_button_press_event()
            # if right button is pressed
            elif event == 'RightButtonPressEvent':
                self.mouse_status |= MOUSE.RIGHT
                self.right_button_press_event()

        elif 'ReleaseEvent' in event: # 
            # if left button is released
            if event == 'LeftButtonReleaseEvent':
                self.left_button_release_event()
            # if right button is released
            if event == 'RightButtonReleaseEvent':
                self.right_button_release_event()
            # end of an event cycle - reset mouse status
            self.mouse_status = 0

        elif event == 'MouseMoveEvent':
            self.mouse_status = self.mouse_status | MOUSE.MOVED
            self.mouse_move_event()


    # Interactor callbacks

    def left_button_press_event(self):
        return self.style.OnLeftButtonDown()

    def left_button_release_event(self):
        return self.style.OnLeftButtonUp()

    def right_button_press_event(self):
        return self.style.OnRightButtonDown()

    def right_button_release_event(self):
        return self.style.OnRightButtonUp()

    def mouse_move_event(self):
        return self.style.OnMouseMove()


class Selector(Window):


    DEFAULT_PICKER_CLASS = vtkCellPicker
    DEFAULT_PICKER_TOLERANCE = 1e-6


    def __init__(self, **kwargs):
        super().__init__()
        self.attach_picker(**{k:v for k,v in kwargs.items() if k.startswith('picker')})
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


    def add_pick_polydata(self, polyd):
        
        cc = vtkCurvatures()
        cc.SetInputData(polyd)
        cc.SetCurvatureTypeToMean()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cc.GetOutputPort())
        mapper.SetScalarModeToUsePointData()
        mapper.SetLookupTable(self.lut)
        mapper.SetScalarRange(-.5,.5)
        mapper.SetArrayName('Mean_Curvature')
        actor = vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)
        self.picker.AddPickList(actor)
    
        return None

    def add_show_polydata(self, polyd):

        cc = vtkCurvatures()
        cc.SetInputData(polyd)
        cc.SetCurvatureTypeToMean()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cc.GetOutputPort())
        mapper.SetScalarModeToUsePointData()
        mapper.SetLookupTable(self.lut)
        mapper.SetScalarRange(-.5,.5)
        mapper.SetArrayName('Mean_Curvature')
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(.5)
        self.renderer.AddActor(actor)
    
        return None


    def get_pick_props(self):
        l = self.picker.GetPickList()
        props = [vtkProp.SafeDownCast(l.GetItemAsObject(i)) for i in range(l.GetNumberOfItems())]
        return props


    def get_pick_mappers(self):
        props = self.get_pick_props()
        mapps = []
        for p in props:
            act = vtkActor.SafeDownCast(p)
            if act:
                mapps.append(act.GetMapper())
            else:
                mapps.append(None)


    def get_pick_properties(self):
        props = self.get_pick_props()
        pptys = []
        for p in props:
            act = vtkActor.SafeDownCast(p)
            if act:
                pptys.append(act.GetProperty())
            else:
                pptys.append(None)


    def get_pick_polydata(self):
        return [m.GetInput() for m in self.get_pick_mappers()]


    def get_selection_indices(self):
        pass


class PolygonalSurfacePointSelector(Selector):
    '''a simple instruction on user interactions:

    simple mouse click to select or deselect a point

    hold ctrl key and click to:
        if a point is selected, modify
        if no point is selected, add a new one

    press "Delete" key to delete the current selection

    programmer's note:
    follow the example below
    ```
        sel = PolygonalSurfacePointSelector()
        sel.add_pick_polydata(some_vtkPolyData_instance) # add surface to pick from
        sel.initialize(
            mode=MODE.SELECT|MODE.ADD, 
            dict_of_named_points={'Sella'=[1,2,3],'Porion'=[4,5,6]})  # set the mode and add initial points
        self.start()
    ```
    each action (select, modify, add, delete) can be enabled or disable by setting mode

    '''
    DEFAULT_GLYPH_COLOR = (0.8, 0.0, 0.0)
    SELECTED_GLYPH_COLOR = (0.5, 1.0, 0.5)


    def initialize(self, mode=MODE.MODIFY, dict_of_named_points={}):

        self.mode = mode
        self.selection_names = list(dict_of_named_points.keys())
        self.selection_points = vtkPoints()
        self.current_id = None

        for c in dict_of_named_points.values():
            self.selection_points.InsertNextPoint(c)

        self.selection_points.Modified()
        
        inp = vtkPointSet()
        inp.SetPoints(self.selection_points)
        src = vtkSphereSource()
        src.SetRadius(1.0)
        src.Update()

        # set up glyph3d
        glyph_filter = vtkGlyph3D()
        glyph_filter.SetSourceConnection(src.GetOutputPort())
        glyph_filter.SetInputData(inp)
        glyph_filter.GeneratePointIdsOn()
        glyph_filter.Update()

        # attach color
        self.lookup_table = vtkColorTransferFunction()
        self.lookup_table.Build()
        # self.lookup_table.IndexedLookupOn()
        for x in range(len(dict_of_named_points)):
            self.lookup_table.AddRGBPoint(x,*self.DEFAULT_GLYPH_COLOR)

        # display these points
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph_filter.GetOutputPort())
        mapper.SetLookupTable(self.lookup_table)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray('InputPointIds')

        prop = vtkActor()
        prop.SetMapper(mapper)

        # set up picker for these points
        # not the same as self.picker
        # self.picker picks from the surface
        # self.glyph_picker picks from point glyphs
        self.glyph_picker = vtkCellPicker()
        self.glyph_picker.SetTolerance(self.DEFAULT_PICKER_TOLERANCE)
        self.glyph_picker.InitializePickList()
        self.glyph_picker.SetPickFromList(True)
        self.glyph_picker.AddPickList(prop)
        self.renderer.AddActor(prop)

        # current selection legend
        self.legend = vtkTextActor()
        self.legend.SetPosition(0,0)
        self.legend.SetInput('No Selection')
        self.legend.GetTextProperty().BoldOn()
        self.legend.GetTextProperty().SetFontSize(36)
        self.legend.GetTextProperty().SetColor(.4,0,0)
        # self.legend.GetTextProperty().SetJustificationToCentered()
        self.legend.PickableOff()
        self.renderer.AddActor(self.legend)


    def _next_name(self):
        i = 1
        while str(i) in self.selection_names:
            i = i+1
        return str(i)


    def left_button_press_event(self):

        pos = self.style.GetInteractor().GetEventPosition()
        self.glyph_picker.Pick(pos[0], pos[1], 0, self.style.GetDefaultRenderer())
        
        # following actions are mutually exclusive
        # but for simplicity, they are not written in if-elif-else

        # if ctrl is not pressed, and a landmark is hit, possibly select the landmark
        if not self.mouse_status & MOUSE.CTRL and self.mode.allows_select():

            if self.glyph_picker.GetCellId() >= 0:
                self._select()
            else:
                self._deselect()


        elif self.mouse_status & MOUSE.CTRL:
            # if a landmark is already selected and modification is enabled
            if self.current_id is not None and self.mode.allows_modify():

                # then update its location and end interaction
                self._modify()
                return None

            # if a landmark is not selected and addition is enabled
            if self.current_id is None and self.mode.allows_add():

                # then add a new point and end interaction
                self._add()
                return None

        # continue with a normal press
        return super().left_button_press_event()


    def _select(self):

        if self.current_id is not None:
            self.lookup_table.AddRGBPoint(self.current_id, *self.DEFAULT_GLYPH_COLOR)

        glyph_polyd = self.glyph_picker.GetDataSet()

        # find out which landmark it is
        ids = glyph_polyd.GetPointData().GetArray('InputPointIds')
        id_ = vtkIdTypeArray.SafeDownCast(ids).GetTuple(self.glyph_picker.GetPointId())[0]
        id_ = int(id_)

        self.lookup_table.AddRGBPoint(id_, *self.SELECTED_GLYPH_COLOR)
        self.current_id = id_

        self.legend.SetInput(self.selection_names[self.current_id])
        self.render_window.Render()

        return None


    def _deselect(self):
        if self.current_id is not None:
            self.lookup_table.AddRGBPoint(self.current_id, *self.DEFAULT_GLYPH_COLOR)

        self.current_id = None
        self.legend.SetInput('No Selection')
        self.render_window.Render()

        return None


    def _modify(self):
        pos = self.style.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, self.style.GetDefaultRenderer())

        if self.picker.GetCellId() >= 0:
            self.selection_points.SetPoint(self.current_id, self.picker.GetPickPosition())
            self.selection_points.Modified()

        self.render_window.Render()

        return None


    def _add(self):
        pos = self.style.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, self.style.GetDefaultRenderer())

        if self.picker.GetCellId() >= 0:
            self.selection_names.append(self._next_name())
            self.selection_points.InsertNextPoint(self.picker.GetPickPosition())
            self.selection_points.Modified()
            self.lookup_table.AddRGBPoint(self.selection_points.GetNumberOfPoints()-1,*self.DEFAULT_GLYPH_COLOR)
            self.legend.SetInput(f'point {self.selection_names[-1]} added')
            self.render_window.Render()

        return None


    def _delete(self):
        if not self.mode.allows_delete() or self.current_id is None:
            return None
        
        name = self.selection_names[self.current_id]
        self.selection_points.GetData().RemoveTuple(self.current_id)
        self.selection_points.Modified()
        del self.selection_names[self.current_id]
        self.lookup_table.AddRGBPoint(self.current_id, *self.DEFAULT_GLYPH_COLOR)
        self.lookup_table.RemovePoint(self.selection_points.GetNumberOfPoints()-1)
        self.current_id = None
        self.legend.SetInput(f'{name} deleted')
        self.render_window.Render()
        return None
    

    def key_press_event(self, key):

        if key == "Escape":
            return self._deselect()

        elif key.startswith('Control'):
            if self.current_id is None and self.mode.allows_add():
                self.legend.SetInput(f'Adding Point {self._next_name()}')
            elif self.current_id is not None and self.mode.allows_modify():
                self.legend.SetInput(f'Modifying {self.selection_names[self.current_id]}')
            self.render_window.Render()
            return None

        elif key == 'Delete':
            return self._delete()
        
        return super().key_press_event(key)

            
    def key_release_event(self, key):

        if key.startswith('Control'):
            if self.current_id is None:
                self.legend.SetInput('No Selection')
            elif self.current_id is not None:
                self.legend.SetInput(self.selection_names[self.current_id])
            self.render_window.Render()
            return None
        
        return super().key_release_event(key)


class PolygonalSurfaceNodeSelector(Selector):

    def initialize(self, pick_surf:vtkPolyData, other_surf:vtkPolyData):

        self.add_pick_polydata(pick_surf)
        self.add_show_polydata(other_surf)

        index_id = vtkIdTypeArray()
        self.selection = index_id
        index_id.SetName('Index')
        index_id.SetNumberOfTuples(pick_surf.GetNumberOfPoints())
        index_id.SetNumberOfComponents(1)
        index_id.Fill(0)
        pick_surf.GetPointData().AddArray(index_id)

        src = vtkSphereSource()
        src.SetRadius(.5)
        src.Update()

        glyph_mapper = vtkGlyph3DMapper()
        glyph_mapper.SetSourceIndexArray('Index')
        glyph_mapper.SourceIndexingOn()
        glyph_mapper.SetLookupTable(self.lut)
        glyph_mapper.SetScalarRange(-.5,.5)
        self.current_id = None

        glyph_mapper.SetSourceData(0, vtkPolyData())
        glyph_mapper.SetSourceConnection(1, src.GetOutputPort())
        glyph_mapper.SetInputData(pick_surf)
        glyph_mapper.Update()

        glyph_actor = vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        self.renderer.AddActor(glyph_actor)

        return None


    def _select(self):

        pos = self.style.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, self.style.GetDefaultRenderer())
        id = self.picker.GetPointId()
        if id >= 0:
            self.current_id = id
            tup = self.selection.GetTuple(id)
            if not tup[0]:
                self.selection.SetTuple(id, np.array([1], dtype=np.int64))
                self.selection.Modified()
                self.render_window.Render()
        return None
        

    def _deselect(self):

        self.current_id = None
        self.selection.Fill(0)
        self.selection.Modified()
        self.render_window.Render()
        return None
        

    def _modify(self):
        print('modify')
        return None


    def left_button_press_event(self):
        if self.mouse_status & MOUSE.CTRL:
            return self._select()
        return super().left_button_press_event()


    def mouse_move_event(self):
        if self.mouse_status & MOUSE.CTRL and self.mouse_status & MOUSE.LEFT:
            return self._select()
        return super().mouse_move_event()


    def key_press_event(self, key):
        if key == 'Escape':
            return self._deselect()
        elif key == 'space':
            return self._modify()

        return super().key_press_event(key)


