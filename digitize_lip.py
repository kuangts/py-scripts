import os, csv
from itertools import chain
from vtk import vtkMatrix4x4
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkTextActor,    
    vtkProperty,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)



labels = [
    'Ss',
    'Ls',
    'Cph-R',
    'Cph-L',
    'Stm',
    'Ch-R',
    'Ch-L',
    'Li',
]


class LandmarkActor(vtkActor):
    def __init__(self, label, coord):
        super().__init__()
        self.source = vtkSphereSource()
        self.source.SetCenter(coord)
        self.source.SetRadius(1)
        self.mapper = vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.source.GetOutputPort())
        self.SetMapper(self.mapper)
        self.GetProperty().SetColor(colors.GetColor3d('Green'))
        self.GetProperty().SetPointSize(10)
        self.label = label
        self.coord = coord

    def Select(self):
        self.GetProperty().SetColor(colors.GetColor3d('Green'))
        self.mapper.Update()

    def Deselect(self):
        self.GetProperty().SetColor(colors.GetColor3d('Tomato'))
        self.mapper.Update()

    @property
    def coord(self):
        return self.source.GetCenter()

    @coord.setter
    def coord(self, _c):
        self.source.SetCenter(_c)
        self.mapper.Update()


class Digitizer():

    def __init__(self):
        self.lmk = []
        self.docket = []
        global colors
        colors = vtkNamedColors()

        # flying edge to generate model from image
        reader = vtkNIFTIImageReader()
        skin_extractor = vtkFlyingEdges3D()
        skin_extractor.SetInputConnection(reader.GetOutputPort())
        skin_extractor.SetValue(0, 500)
        skin_mapper = vtkPolyDataMapper()
        skin_mapper.SetInputConnection(skin_extractor.GetOutputPort())
        skin_mapper.ScalarVisibilityOff()
        skin = vtkActor()
        skin.SetMapper(skin_mapper)
        skin.GetProperty().SetDiffuseColor(colors.GetColor3d('peachpuff'))
        back_prop = vtkProperty()
        back_prop.SetDiffuseColor(colors.GetColor3d('peachpuff3'))
        skin.SetBackfaceProperty(back_prop)

        # create window and render
        renderer = vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('paleturquoise'))
        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(renderer)
        ren_win.SetSize(640, 480)
        ren_win.SetWindowName('Lip Landmarks')
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        style = vtkInteractorStyleTrackballCamera()
        style.SetDefaultRenderer(renderer)
        iren.SetInteractorStyle(style)
        renderer.AddActor(skin)

        # Interactor callbacks
        style.rotated = False # flag for dragging
        style.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        style.AddObserver('LeftButtonReleaseEvent', self.left_button_release_event)
        style.AddObserver('MouseMoveEvent', self.mouse_move_event)
        style.AddObserver('KeyPressEvent', self.key_press_event)

        status = vtkTextActor()
        status.SetPosition2(10, 40)
        status.GetTextProperty().SetFontSize(16)
        status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        renderer.AddActor2D(status)

        self.reader = reader
        self.ren_win = ren_win
        self.iren = iren
        self.status = status
        self.renderer = renderer

    def start(self):
        self.ren_win.Render()
        self.iren.Initialize()
        self.iren.Start()

    def quit(self):
        self.ren_win.Finalize()
        self.iren.TerminateApp()
        del self.ren_win, self.iren

    # Interactor callbacks
    def key_press_event(self, obj, key):
        key = obj.GetInteractor().GetKeySym()
        print(key)
        if key=='space':
            self.next_lmk()
        elif key=='r':
            self.load()
        elif key=='Return':
            self.next_case()
        elif key=='d' or key=='Delete':
            self.del_lmk()
        elif key=='BackSpace':
            self.previous_lmk()
        elif key=='s':
            self.save()

    def left_button_press_event(self, obj, event):
        obj.rotated = False
        obj.OnLeftButtonDown()

    def mouse_move_event(self, obj, event):
        obj.rotated = True
        obj.OnMouseMove()

    def left_button_release_event(self, obj, event):
        # Get the location of the click (in window coordinates)
        if not obj.rotated:
            pos = obj.GetInteractor().GetEventPosition()
            picker = vtkCellPicker()
            picker.SetTolerance(0.0005)
            # Pick from this location.
            picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
            points = picker.GetPickedPositions()
            actors = picker.GetActors()
            for i in range(points.GetNumberOfPoints()):
                actor = actors.GetItemAsObject(i)
                if actor not in self.lmk:
                    self.add_lmk(points.GetPoint(i))
                    break
        obj.OnLeftButtonUp()

    def refresh(self, text=None):
        if text is None:
            text = '\n'.join((self.current_case, self.print_lmk(self.current_label)))
        self.status.SetInput(text)
        self.ren_win.Render()

    def print_lmk(self, label=None, coord=None):
        if label is None:
            return '\n'.join(self.print_lmk(x.label, x.coord) for x in self.lmk)
        elif coord is None:
            lmk = self.get_lmk(self.current_label)
            if lmk:
                return self.print_lmk(lmk.label, lmk.coord)
            else:
                return f'{label} not digitized'
        else:
            return f'{label:>10}: {[round(x,3) for x in coord]}'

    def get_lmk(self, label):
        acts = list(filter(lambda x:x.label==label, self.lmk))
        if acts:
            return acts[0]
        else:
            return None

    def add_lmk(self, coord):
        lmk = self.get_lmk(self.current_label)
        if lmk:
            lmk.coord = coord
        else:
            lmk = LandmarkActor(self.current_label, coord)
            self.lmk.append(lmk)
            self.renderer.AddActor(lmk)
            lmk.Select()

        self.refresh()

    def del_lmk(self):
        lmk = self.get_lmk(self.current_label) 
        if lmk:
            self.renderer.RemoveActor(lmk)
            self.lmk.remove(lmk)

        self.refresh()

    def next_lmk(self):
        lmk = self.get_lmk(self.current_label)
        if lmk:
            lmk.Deselect()
        ind = labels.index(self.current_label) + 1
        if ind < len(labels):
            self.current_label = labels[ind]
            lmk = self.get_lmk(self.current_label)
            if lmk:
                lmk.Select()
            self.refresh()
        else:
            self.refresh('end of the list')

    def previous_lmk(self):
        lmk = self.get_lmk(self.current_label)
        if lmk:
            lmk.Deselect()
        ind = labels.index(self.current_label) - 1
        if ind >= 0:
            self.current_label = labels[ind]
            lmk = self.get_lmk(self.current_label)
            if lmk:
                lmk.Select()
            self.refresh()
        else:
            self.refresh('beginning of the list')
        
    def add_cases(self, list_of_cases):
        # make sure no duplicate
        self.docket += list_of_cases

    def next_case(self):
        ind = self.docket.index(self.current_case) + 1
        if ind >= len(self.docket):
            self.refresh('no more cases')
            return
        self.load(self.docket[ind])
        self.refresh()

    def save(self, lmk=None, savepath=None):
        if lmk is None:
            lmk = {x.label:x.coord for x in self.lmk}
        if savepath is None:
            filename = self.current_case
            savepath = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace('.nii.gz','-lip.csv'))
        with open(savepath, 'w', newline='') as f:
            csv.writer(f).writerows(zip(list(lmk.keys()),*zip(*list(lmk.values()))))
        self.refresh(f'written to {savepath}')
        
    def load(self, filename=None):
        if filename is None:
            if hasattr(self, 'current_case'):
                filename = self.current_case
            else:
                self.current_case = self.docket[0]
                filename = self.current_case
        else:
            self.current_case = filename
        self.reader.SetFileName(filename)
        self.current_label = labels[0]
        for x in self.lmk:
            self.renderer.RemoveActor(x)
        self.lmk = []
        self.reader.Update()
        self.refresh()


if __name__ == '__main__':
    # root = sys.argv[1]
    # all_cases = glob.glob(os.path.join(root, '*', '*.nii.gz'))
    # cases_keep = []
    # for i,c in enumerate(all_cases):
    #     if not os.path.isfile(c.replace('.nii.gz','-lip.csv')):
    #         cases_keep.append(c)
    d = Digitizer()
    d.add_cases( [r'C:\Users\tmhtxk25\Box\RPI\data\pre-post-paired-unc40-send-1122\n0001\20110425-pre.nii.gz'] )
    d.load()
    d.start()

