import sys, os, csv, glob
import numpy as np
from tkinter import Tk
Tk().withdraw()
from tkinter.filedialog import asksaveasfile
from vtk import vtkMatrix4x4
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkCommonDataModel import vtkPointSet
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkLinearTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
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
    vtkRenderer
)
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk



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

nifti = r'C:\data\pre-post-paired-40-send-1122\n0009\20100426-pre.nii.gz'
lmk_csv = r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\n09-soft.csv'
coord = r'C:\data\pre-post-paired-40-send-1122\n0009\skin_landmark.txt'

class CustomInteractorStyle(vtkInteractorStyleTrackballCamera):
    def __init__(self, digitizer):
        self.pick_mode = True # flag for dragging
        self.AddObserver('RightButtonPressEvent', digitizer.right_button_press_event)
        self.AddObserver('RightButtonReleaseEvent', digitizer.right_button_release_event)
        self.AddObserver('MouseMoveEvent', digitizer.mouse_move)
        self.AddObserver('KeyPressEvent', digitizer.key_press_event)
    def OnRightButtonDown(self):
        super().OnLeftButtonDown()
    def OnRightButtonUp(self):
        super().OnLeftButtonUp()
    def OnLeftButtonDown(self):
        super().OnRightButtonDown()
    def OnLeftButtonUp(self):
        super().OnRightButtonUp()


class Digitizer():

    def __init__(self, nifti, lmk_csv, coord):

        global colors
        colors = vtkNamedColors()

        # flying edge to generate model from image
        reader = vtkNIFTIImageReader()
        reader.SetFileName(nifti)
        reader.Update()
        T = vtkMatrixToLinearTransform()
        m = vtkMatrix4x4()
        m.DeepCopy(reader.GetSFormMatrix())
        t = np.eye(4)
        m.DeepCopy(t.ravel(), m)
        t[0,3] = t[0,3]*t[0,0]
        t[1,3] = t[1,3]*t[1,1]
        t[0,0] = 1
        t[1,1] = 1
        m.DeepCopy(t.ravel())
        T.SetInput(m) 

        skin_extractor = vtkFlyingEdges3D()
        skin_extractor.SetInputConnection(reader.GetOutputPort())
        skin_extractor.SetValue(0, 500)
        transformer = vtkTransformPolyDataFilter()
        transformer.SetTransform(T)
        transformer.SetInputConnection(skin_extractor.GetOutputPort())
        transformer.Update()
        skin_mapper = vtkPolyDataMapper()
        skin_mapper.SetInputConnection(transformer.GetOutputPort())
        skin_mapper.ScalarVisibilityOff()
        skin = vtkActor()
        skin.SetMapper(skin_mapper)
        skin.PickableOff()
        skin.GetProperty().SetDiffuseColor(colors.GetColor3d('peachpuff'))

        # create window and render
        renderer = vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('paleturquoise'))
        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(renderer)
        ren_win.SetSize(640, 480)
        ren_win.SetWindowName('Lip Landmarks')
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        style = CustomInteractorStyle(self)
        style.SetDefaultRenderer(renderer)
        iren.SetInteractorStyle(style)
        renderer.AddActor(skin)

        status = vtkTextActor()
        status.SetPosition2(10, 40)
        status.GetTextProperty().SetFontSize(16)
        status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        renderer.AddActor2D(status)

        self.reader = reader
        self.lmk = {}
        self.ren_win = ren_win
        self.iren = iren
        self.ren = renderer
        self.status = status
        self.skin = skin

        self.mapping = {}


        self.coord = {i:c for i,c in enumerate(np.genfromtxt(coord)) if c.any()}
        self.lmk1 = []
        for i,c in self.coord.items():
            self.lmk1.append(Landmark(str(i),c,Color=(1,1,0)))
            self.lmk1[-1].SetRenderer(self.ren)

        with open(lmk_csv, 'r', newline='') as f:
            self.lmk = {x[0]:(np.asarray([[*x[1:],1.]],dtype=float)@t.T)[0,:-1] for x in csv.reader(f)}
        self.lmk2 = []
        for l,c in self.lmk.items():
            self.lmk2.append(Landmark(l,c))
            self.lmk2[-1].SetRenderer(self.ren)

        self.selection1 = None
        self.selection2 = None

        self.ren_win.Render()
        self.iren.Initialize()
        self.iren.Start()

    @property
    def index(self):
        try:
            return self.lmk1[self.id1].label
        except:
            return None

    @property
    def label(self):
        try:
            return self.lmk2[self.id2].label
        except:
            return None

    def key_press_event(self, obj, event):
        key = self.iren.GetKeySym()
        if key=='Return':
            self.confirm()
        elif key=='t':
            if self.skin.GetVisibility():
                self.skin.SetVisibility(False)
            elif not self.skin.GetVisibility():
                self.skin.SetVisibility(True)
        elif key == 's':
            if self.iren.GetControlKey():
                self.save()
        elif key == 'b':
            pass # self.bone

        self.ren_win.Render()

                
    def right_button_press_event(self, obj, event):
        obj.pick_mode = True
        obj.OnRightButtonDown()

    def mouse_move(self, obj, event):
        obj.pick_mode = False
        obj.OnMouseMove()

    def right_button_release_event(self, obj, event):
        if obj.pick_mode:
            pos = self.iren.GetEventPosition()
            picker = vtkCellPicker()
            picker.SetTolerance(0.0005)
            picker.Pick(pos[0], pos[1], 0, self.ren)
            
            if picker.GetCellId() == -1:
                if hasattr(self,'id1') and self.id1:
                    self.lmk1[self.id1].Deselect()
                if hasattr(self, 'id2') and self.id2:
                    self.lmk2[self.id2].Deselect()
                self.id1 = None
                self.id2 = None
            else:
                actors = picker.GetActors()
                for i in range(actors.GetNumberOfItems()):
                    a = actors.GetItemAsObject(i)
                    try:
                        id1 = [x.coord_actor for x in self.lmk1].index(a)
                        if hasattr(self,'id1') and self.id1:
                            self.lmk1[self.id1].Deselect()
                        self.id1 = id1
                        self.lmk1[self.id1].Select()
                    except:
                        # print('Not in set 1')
                        pass
                    try:
                        id2 = [x.coord_actor for x in self.lmk2].index(a)
                        if hasattr(self, 'id2') and self.id2:
                            self.lmk2[self.id2].Deselect()
                        self.id2 = id2
                        self.lmk2[self.id2].Select()
                    except:
                        # print('Not in set 2')
                        pass

        # Forward events
        obj.pick_mode = True
        obj.OnRightButtonUp()

    def confirm(self):
        self.mapping[self.index] = self.label
        print(f'{self.index}-->{self.label}')
        self.lmk1[self.id1].Remove()        
        self.lmk2[self.id2].Remove()
        self.ren_win.Render()

    def save(self):
        with asksaveasfile() as f:
            writer = csv.writer(f)
            for k,v in self.mapping.items():
                writer.writerow([k,v])


class Landmark:
    SelectionColor = [0,1,0]
    Color = [1,0,0]
    def __init__(self, label, coord, **kwargs):
        self.coord = coord
        self.label = label
        for k,v in kwargs.items():
            setattr(self, k, v)
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
        self.coord_actor = act
        self.label_actor = txt
    def SetRenderer(self, ren):
        self.renderer = ren
        ren.AddActor(self.coord_actor)
        ren.AddActor(self.label_actor)
    def Remove(self):
        self.renderer.RemoveActor(self.coord_actor)
        self.renderer.RemoveActor(self.label_actor)
    def Select(self):
        self.coord_actor.GetProperty().SetColor(*self.SelectionColor)
        self.label_actor.GetTextProperty().SetColor(*self.SelectionColor)
    def Deselect(self):
        self.coord_actor.GetProperty().SetColor(*self.Color)
        self.label_actor.GetTextProperty().SetColor(*self.Color)


def main(argv):
    Digitizer(nifti, lmk_csv, coord)

if __name__ == '__main__':
    main(sys.argv[1:])

