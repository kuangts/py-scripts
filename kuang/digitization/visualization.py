# rewrite with this:
# https://kitware.github.io/vtk-examples/site/Cxx/Interaction/MoveAGlyph/


from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
      
class vtkLandmark:

    Color = (1,0,0)
    HighlightColor = (0,1,0)

    def __init__(self, label:str, coord=(float('nan'),)*3, **kwargs):
        src = vtkSphereSource()
        src.SetCenter(*coord)
        src.SetRadius(1)
        map = vtkPolyDataMapper()
        map.SetInputConnection(src.GetOutputPort())
        act = vtkActor()
        act.SetMapper(map)
        act.GetProperty().SetColor(*self.Color)
        txt = vtkBillboardTextActor3D()
        txt.SetPosition(*coord)
        txt.SetInput(label)
        txt.GetTextProperty().SetFontSize(24)
        txt.GetTextProperty().SetJustificationToCentered()
        txt.GetTextProperty().SetColor(*self.Color)
        txt.PickableOff()

        self.sphere = src
        self.sphere_actor = act
        self.label_actor = txt
        self.sphere_actor.prop_name = 'ldmk'
        self.sphere_actor.parent = self

        for k,v in kwargs.items():
            setattr(self, k, v)

    @property
    def label(self):
        return self.label_actor.GetInput().strip()

    def move_to(self, new_coord):
        self.sphere.SetCenter(*new_coord)
        self.sphere.Update()
        self.label_actor.SetPosition(*new_coord)

    def set_renderer(self, ren):
        self.renderer = ren
        ren.AddActor(self.sphere_actor)
        ren.AddActor(self.label_actor)

    def remove(self):
        self.renderer.RemoveActor(self.sphere_actor)
        self.renderer.RemoveActor(self.label_actor)
        del self.sphere_actor, self.label_actor, self.sphere

    def refresh(self):
        if hasattr(self, 'selected') and self.selected:
            self.sphere_actor.GetProperty().SetColor(*self.HighlightColor)
            self.label_actor.GetTextProperty().SetColor(*self.HighlightColor)
        else:
            self.sphere_actor.GetProperty().SetColor(*self.Color)
            self.label_actor.GetTextProperty().SetColor(*self.Color)

        self.sphere.Update()

    def select(self):
        self.selected = True
        self.refresh()

    def deselect(self):
        self.selected = False
        self.refresh()



