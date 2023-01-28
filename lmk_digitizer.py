import sys, os, csv, glob, math
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
import landmark
from landmark import LandmarkDict

computed = [
    "Gb'",
    "N'",
    "Zy'-R",
    "Zy'-L",
    "CM",
    "Pog'",
    "Me'",
    "Go'-R",
    "Go'-L",
]

digitized = [
    "En-R",
    "En-L",
    "Ex-R",
    "Ex-L",
    "Prn",
    "C",
    "Sn",
    "Ls",
    "Stm-U",
    "Stm-L",
    "Ch-R",
    "Ch-L",
    "Li",
    "Sl",
]

lib_computed = landmark.library.filter(lambda x:x.Name in computed)
lib_digitized = landmark.library.filter(lambda x:x.Name in digitized)

lmk = LandmarkDict()

colors = vtkNamedColors()

class Digitizer():

    def __init__(self):
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
        style.pick_mode = True # flag for dragging
        style.AddObserver('RightButtonPressEvent', self.right_button_press_event)
        style.AddObserver('RightButtonReleaseEvent', self.right_button_release_event)
        style.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        style.AddObserver('LeftButtonReleaseEvent', self.left_button_release_event)
        style.AddObserver('MouseMoveEvent', self.mouse_move)
        style.AddObserver('KeyPressEvent', self.key_press_event)
        style.picker = vtkCellPicker()
        style.picker.SetTolerance(0.0005)
        self.ren = renderer
        self.ren_win = ren_win
        self.iren = iren
        self.style = style


    def setup(self, nifti_file, ldmk_file):

        reader = vtkNIFTIImageReader()
        reader.SetFileName(nifti_file)
        reader.Update()

        skin_extractor = vtkFlyingEdges3D()
        skin_extractor.SetInputConnection(reader.GetOutputPort())
        skin_extractor.SetValue(0, 324)
        skin_mapper = vtkPolyDataMapper()
        skin_mapper.SetInputConnection(skin_extractor.GetOutputPort())
        skin_mapper.ScalarVisibilityOff()
        self.skin = vtkActor()
        self.skin.SetMapper(skin_mapper)
        self.skin.PickableOn()
        self.skin.GetProperty().SetDiffuseColor(colors.GetColor3d('peachpuff'))
        self.skin.SetVisibility(True)
        self.skin.prop_name = 'skin'
        self.ren.AddActor(self.skin)

        bone_extractor = vtkFlyingEdges3D()
        bone_extractor.SetInputConnection(reader.GetOutputPort())
        bone_extractor.SetValue(0, 1150)
        bone_mapper = vtkPolyDataMapper()
        bone_mapper.SetInputConnection(bone_extractor.GetOutputPort())
        bone_mapper.ScalarVisibilityOff()
        self.bone = vtkActor()
        self.bone.SetMapper(bone_mapper)
        self.bone.PickableOn()
        self.bone.GetProperty().SetDiffuseColor(colors.GetColor3d('peachpuff'))
        self.bone.SetVisibility(False)
        self.bone.prop_name = 'bone'
        self.ren.AddActor(self.bone)

        self.status = vtkTextActor()
        self.status.SetPosition2(10, 40)
        self.status.GetTextProperty().SetFontSize(16)
        self.status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        self.ren.AddActor2D(self.status)

        self.vtk_lmk = [] # a collection of landmark props
        self.file = ldmk_file
        if os.path.isfile(self.file):
            if self.file.endswith('.xlsx'):
                self.lmk = LandmarkDict.from_excel(self.file)
            elif self.file.endswith('.cass'):
                self.lmk = LandmarkDict.from_excel(self.file)
            else:
                self.lmk = LandmarkDict.read(self.file)
        else:
            self.lmk = LandmarkDict()

        self.lmk = self.lmk.sort_by([*computed,  *digitized])
        print(self.lmk)

        for k,v in self.lmk.items():
            if k in computed:
                l = landmark.vtkLandmark(k, v, Color=(1,0,1), HighlightColor=(1,1,0))
            else:
                l = landmark.vtkLandmark(k, v)
            self.vtk_lmk.append(l)
            l.SetRenderer(self.ren)

        self.style.lmk = self.vtk_lmk[0]
        self.update()

    def start(self):
        d.ren_win.Render()
        d.iren.Initialize()
        d.iren.Start()


    ######   interactor callbacks   ######
    def key_press_event(self, obj, event):
        key = self.iren.GetKeySym()
        if key=='Return':
            self.next()
        elif key=='t':
            if self.skin.GetVisibility():
                self.skin.SetVisibility(False)
            elif not self.skin.GetVisibility():
                self.skin.SetVisibility(True)
        elif key == 's':
            if self.iren.GetControlKey():
                self.save()
        elif key == 'b':
            if self.bone.GetVisibility():
                self.bone.SetVisibility(False)
            elif not self.bone.GetVisibility():
                self.bone.SetVisibility(True)

        self.ren_win.Render()

    def left_button_press_event(self, obj, event):
        obj.pick_mode = True
        obj.left_button = True
        obj.OnLeftButtonDown()

    def left_button_release_event(self, obj, event):
        if obj.pick_mode and obj.left_button:
            self.pick_select(obj, event)
        # Forward events
        obj.pick_mode = True
        obj.left_button = False
        obj.OnLeftButtonUp()

    def right_button_press_event(self, obj, event):
        obj.pick_mode = True
        obj.right_button = True

    def right_button_release_event(self, obj, event):
        if obj.pick_mode and obj.right_button:
            self.pick_place(obj, event)
        # Forward events
        obj.pick_mode = True
        obj.right_button = False

    def mouse_move(self, obj, event):
        obj.pick_mode = False
        obj.OnMouseMove()

    def pick_select(self, obj, event):
        pos = self.iren.GetEventPosition()
        obj.picker.Pick(pos[0], pos[1], 0, self.ren)
        if obj.picker.GetCellId() == -1:
            return
        else:
            prop = obj.picker.GetProp3D()
            if prop.prop_name == 'ldmk':
                if hasattr(obj,'lmk'):
                    obj.lmk.Deselect()
                obj.lmk = prop.parent
                obj.lmk.Select()
            self.update()

    def pick_place(self, obj, event):
        pos = self.iren.GetEventPosition()
        obj.picker.Pick(pos[0], pos[1], 0, self.ren)
        prop = obj.picker.GetProp3D()
        if obj.picker.GetCellId() == -1:
            return
        else:
            prop = obj.picker.GetProp3D()
            if prop.prop_name != 'ldmk':
                self.lmk[obj.lmk.label] = obj.picker.GetPickPosition()
                obj.lmk.MoveTo(self.lmk[obj.lmk.label])
            self.update()
    
    def update(self, text=''):
        if not text:
            if hasattr(self.style, 'lmk'):
                l = self.style.lmk.label
                entr = landmark.library.find(Name=l)
                text = '\n'.join((f'{l} - {entr.Category}', f'{self.lmk[l]}', entr.Fullname, entr.Description))
        self.status.SetInput(text)
        self.ren_win.Render()

    def next(self):
        remaining_lmk = [k for k,v in self.lmk.items() if all(math.isnan(x) for x in v)]
        for x in self.vtk_lmk:
            if x.label in remaining_lmk:
                self.style.lmk = x
                self.update()
                break
        else:
            delattr(self.style, 'lmk')
            self.update('No More Landmarks!')

    def save(self):
        with asksaveasfile() as f:
            writer = csv.writer(f)
            for k,v in self.lmk:
                writer.writerow([k,*v])

if __name__ == '__main__':

    d = Digitizer()
    d.setup(
        r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\n09.nii.gz',
        r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\Landmark Comparison (2023-1-19) soft.xlsx'
        )
    d.start()
