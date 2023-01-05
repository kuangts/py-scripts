import sys, os, csv, glob
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



class MouseInteractorStyle(vtkInteractorStyleTrackballCamera):

    def __init__(self, digitizer):
        self.controller = digitizer
        self.should_add_lmk = True # flag for dragging
        self.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        self.AddObserver('LeftButtonReleaseEvent', self.left_button_release_event)
        self.AddObserver('MouseMoveEvent', self.mouse_move)
        self.AddObserver('KeyPressEvent', self.key_press_event)


    def key_press_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        # print(key)
        if key=='space':
            self.controller.next()
        elif key=='r':
            self.controller.load(self.controller.reader.GetFileName())
        elif key=='Return':
            self.controller.load()
        elif key=='BackSpace':
            self.controller.remove_lmk()

    # Catch mouse events
    def left_button_press_event(self, obj, event):
        self.should_add_lmk = True
        # Forward events
        self.OnLeftButtonDown()

    def mouse_move(self, obj, event):
        self.should_add_lmk = False
        super().OnMouseMove()

    def left_button_release_event(self, obj, event):
        if self.should_add_lmk:
            # Get the location of the click (in window coordinates)
            pos = self.GetInteractor().GetEventPosition()
            picker = vtkCellPicker()
            picker.SetTolerance(0.0005)
            # Pick from this location.
            picker.Pick(pos[0], pos[1], 0, self.GetDefaultRenderer())
            if picker.GetCellId() != -1:
                self.controller.add_lmk(picker.GetPickPosition())
        # Forward events
        self.should_add_lmk = True
        self.OnLeftButtonUp()




class Digitizer():

    def __init__(self, man):
        self.manager = man
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
        global renderer, ren_win, iren, style
        renderer = vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('paleturquoise'))
        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(renderer)
        ren_win.SetSize(640, 480)
        ren_win.SetWindowName('Lip Landmarks')
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        style = MouseInteractorStyle(self)
        style.SetDefaultRenderer(renderer)
        iren.SetInteractorStyle(style)
        renderer.AddActor(skin)

        '''
        vtkNew<vtkTextActor> textActor;
        textActor->SetInput("Hello world");
        textActor->SetPosition2(10, 40);
        textActor->GetTextProperty()->SetFontSize(24);
        textActor->GetTextProperty()->SetColor(colors->GetColor3d("Gold").GetData());
        renderer->AddActor2D(textActor);

        '''
        status = vtkTextActor()
        status.SetPosition2(10, 40)
        status.GetTextProperty().SetFontSize(16)
        status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        renderer.AddActor2D(status)

        self.reader = reader
        self.lmk = {}
        self.ren_win = ren_win
        self.iren = iren
        self.status = status

    def start(self):
        self.ren_win.Render()
        self.iren.Initialize()
        self.iren.Start()

    def quit(self):
        self.ren_win.Finalize()
        self.iren.TerminateApp()
        del self.ren_win, self.iren

    def print_lmk(self, label=None, coord=None):
        if label is None:
            for k,v in self.lmk.items():
                self.print_lmk(k,v)
        elif coord is None:
            coord = self.lmk[label] if label in self.lmk else 'not digitized'
            self.print_lmk(label, coord)
        else:
            print(f'{label:>10}: {coord}')

    def add_lmk(self, coord):
        l = self.manager.current_label
        self.print_lmk(l, coord)
        if l in self.lmk:
            self.lmk[l]['coord'] = coord
            self.lmk[l]['source'].SetCenter(coord)
            self.lmk[l]['source'].Update()
        else:
            self.lmk[l] = {}
            src = vtkSphereSource()
            src.SetCenter(coord)
            src.SetRadius(1)
            map = vtkPolyDataMapper()
            map.SetInputConnection(src.GetOutputPort())
            act = vtkActor()
            act.SetMapper(map)
            act.GetProperty().SetColor(colors.GetColor3d('Tomato'))
            act.GetProperty().SetPointSize(10)
            renderer.AddActor(act)

            self.lmk[l] = dict(
                coord=coord,
                source=src,
                actor=act,
            )

    def remove_lmk(self):
        if not len(self.lmk):
            print('nothing to undo')
            return
        l = list(self.lmk.keys())[-1]
        self.manager.current_label = l
        renderer.RemoveActor(self.lmk[l]['actor'])
        del self.lmk[l]
        print('removed ' + l)
        self.status.SetInput(self.manager.status)
        self.ren_win.Render()
        print('-'*88)

    def next(self):
        print('-'*88)
        try:
            self.manager.next_lmk()
            self.status.SetInput(self.manager.status)
            self.ren_win.Render()
        except StopIteration:
            self.write_lmk()
            self.load()
 
    def write_lmk(self, lmk=None, savepath=None):
        if lmk is None:
            lmk = self.lmk
        if savepath is None:
            filename = self.manager.current_case
            savepath = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace('.nii.gz','-lip.csv'))
        with open(savepath, 'w', newline='') as f:
            csv.writer(f).writerows(zip(list(lmk.keys()),*zip(*[x['coord'] for x in lmk.values()])))
        print(f'written to {savepath}')
        print('-'*88)
        
    def load(self, filename=None):
        if filename is None:
            try:
                filename = self.manager.next_case()
                print('loading ' + filename)
                print('-'*88)
            except StopIteration:
                self.quit()
                print("Finished!")

        self.reader.SetFileName(filename)
        self.manager.current_label = labels[0]
        self.status.SetInput(self.manager.status)
        for v in self.lmk.values():
            renderer.RemoveActor(v['actor'])
        self.lmk = {}
        self.reader.Update()
        self.ren_win.Render()
        return self


class Manager:
    def __init__(self, l):
        self._iter = iter(l)
        self.current_case = ''
        self.current_label = labels[0]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.current_label = labels[0]
            self.current_case = self._iter.__next__()           
            return self.current_case
        except StopIteration:
            print('Finished')
        
    def next_case(self):
        return self.__next__()

    def next_lmk(self):
        ind = labels.index(self.current_label) + 1
        if ind < len(labels):
            self.current_label = labels[ind]
            return self.current_label
        else:
            raise StopIteration

    @property
    def status(self):
        return self.current_case + '\n' + self.current_label

if __name__ == '__main__':
    root = sys.argv[1]
    d = Digitizer(
        Manager( glob.glob(os.path.join(root, '*', '*.nii.gz')) )
    )
    d.load()
    d.start()

