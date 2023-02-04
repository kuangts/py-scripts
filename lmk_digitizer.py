import sys, os, csv, glob, math
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
from landmark import vtkLandmark, LandmarkDict, library
import vtk
import numpy as np

lmk_lib = library().jungwook()
computed = list(lmk_lib.computed().field('Name'))
digitized = list((lmk_lib - lmk_lib.computed()).field('Name'))
computed.sort()
digitized.sort()


def get_landmark_list():
    return computed + digitized

colors = vtkNamedColors()

class Digitizer():

    override = True

    @staticmethod
    def load_landmark(file, ordered_list=None):
        # read existing file if any
        if os.path.isfile(file):
            if file.endswith('.xlsx'):
                lmk = LandmarkDict.from_excel(file)
            elif file.endswith('.cass'):
                lmk = LandmarkDict.from_cass(file)
            else:
                lmk = LandmarkDict.read(file)
        else:
            lmk = LandmarkDict()

        # keep only required landmarks
        if ordered_list is not None:
            lmk = lmk.sort_by(ordered_list)

        return lmk

    @staticmethod
    def draw_landmark(lmk, renderer, **kwargs):
        vtk_lmk = [ vtkLandmark(k, v, **kwargs) for k,v in lmk.items() ]
        for l in vtk_lmk:
            l.refresh()
            l.set_renderer(renderer)
        return vtk_lmk

    @staticmethod
    def generate_mdoel(source, threshold, color, **kwargs):
        extractor = vtkFlyingEdges3D()
        extractor.SetInputConnection(source.GetOutputPort())
        extractor.SetValue(*threshold)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(extractor.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOn()
        actor.GetProperty().SetDiffuseColor(color)
        for k,v in kwargs.items():
            setattr(actor, k, v)
        return extractor, actor

    @staticmethod
    def build_model(polyd):
        obbtree = vtk.vtkOBBTree()
        obbtree.SetDataSet(polyd)
        obbtree.BuildLocator()
        return obbtree



    def __init__(self):

        # create window 
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
        style.AddObserver('RightButtonPressEvent', self.right_button_press_event)
        style.AddObserver('RightButtonReleaseEvent', self.right_button_release_event)
        style.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        style.AddObserver('LeftButtonReleaseEvent', self.left_button_release_event)
        style.AddObserver('MouseMoveEvent', self.mouse_move)
        style.AddObserver('KeyPressEvent', self.key_press_event)
        style.pick_mode = True # flag for dragging
        style.picker_left = vtkCellPicker()
        style.picker_left.SetTolerance(0.0005)
        style.picker_right = vtkCellPicker()
        style.picker_right.SetTolerance(0.0005)
        iren.SetInteractorStyle(style)

        self.ren = renderer
        self.ren_win = ren_win
        self.iren = iren
        self.style = style

        # text box for status update, soon to be replaced by PyQt
        self.status = vtkTextActor()
        self.status.SetPosition2(10, 40)
        self.status.GetTextProperty().SetFontSize(16)
        self.status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        self.ren.AddActor2D(self.status)


    def setup(self, nifti_file=None, ldmk_file=None):

        # clear up any existing props
        if hasattr(self, 'vtk_lmk'):
            [ a.remove() for a in self.vtk_lmk ]
        self.vtk_lmk = []
        
        # load landmark
        if ldmk_file is None:
            self.file = None
            self.lmk = LandmarkDict()
        else:
            self.file = ldmk_file
            self.lmk = self.load_landmark(ldmk_file, get_landmark_list())
            nml = self.file.replace('.csv','-normal.csv')
            if nml:
                self.nml = self.load_landmark(nml, get_landmark_list())
                

        if len(self.lmk):

            # landmark props self.vtk_lmk
            vtk_computed = self.draw_landmark(self.lmk.sort_by(computed), self.ren, Color=(1,0,1), HighlightColor=(1,1,0))
            vtk_digitized = self.draw_landmark(self.lmk.sort_by(digitized), self.ren)
            self.vtk_lmk = vtk_computed + vtk_digitized

            # select the first landmark to start
            self.style.selection = self.vtk_lmk[0]

        # load stl models

        # first build the pipeline
        if not hasattr(self, 'reader'):

            # entry point
            self.reader = vtkNIFTIImageReader()

            # build models
            self.skin, self.skin_actor = self.generate_mdoel(
                source=self.reader,
                threshold=(0, 324),
                color=colors.GetColor3d('peachpuff'),
                prop_name='skin'
                )
            self.bone, self.bone_actor = self.generate_mdoel(
                source=self.reader,
                threshold=(0, 1150),
                color=colors.GetColor3d('grey'),
                prop_name='bone'
                )

            self.style.picker_right.InitializePickList()
            self.style.picker_right.AddPickList(self.skin_actor)
            self.style.picker_right.AddPickList(self.bone_actor)
            self.style.picker_right.SetPickFromList(True)

            # add props to renderer
            self.ren.AddActor(self.skin_actor)
            self.ren.AddActor(self.bone_actor)

        if nifti_file is not None:

            # update the input
            self.reader.SetFileName(nifti_file)
            self.reader.Update()

            # update everything else, e.g. status bar, and draw
            self.ren_win.Render()
            self.skin_tree = self.build_model(self.skin.GetOutput())

            self.update()

    def start(self):
        self.iren.Start()




    ######   interactor callbacks   ######
    def key_press_event(self, obj, event):
        key = self.iren.GetKeySym()
        if key=='Return' or key=='space':
            self.next()
        elif key == 't':
            self.skin_actor.SetVisibility(not self.skin_actor.GetVisibility())
        elif key == 'b':
            self.bone_actor.SetVisibility(not self.bone_actor.GetVisibility())
        elif key == 's':
            if self.iren.GetControlKey():
                self.save()
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
        obj.picker_left.Pick(pos[0], pos[1], 0, self.ren)
        if obj.picker_left.GetCellId() == -1:
            return
        else:
            prop = obj.picker_left.GetProp3D()
            # obj.HighlightProp3D(prop)
            if prop.prop_name == 'ldmk':
                if hasattr(obj,'selection'):
                    obj.selection.deselect()
                obj.selection = prop.parent
                obj.selection.select()
            self.update()

    def pick_place(self, obj, event):
        pos = self.iren.GetEventPosition()
        obj.picker_right.Pick(pos[0], pos[1], 0, self.ren)
        if obj.picker_right.GetCellId() == -1:
            return
        self.trace(obj.picker_right.GetPickPosition(), obj.picker_right.GetPickNormal())
        obj.selection.move_to(self.lmk[obj.selection.label])
        self.update()
    
    def update(self, text=''):
        if not text:
            if hasattr(self.style, 'selection'):
                l = self.style.selection.label
                entr = lmk_lib.find(Name=l)
                text = '\n'.join((
                    f'{l} - {entr.Category}',
                    '(' + ', '.join(f'{x:.2f}' for x in self.lmk[l]) + ')',
                    entr.Fullname, 
                    entr.Description))
                    
                
                if l in computed:
                    self.skin_actor.SetVisibility(False)
                    self.bone_actor.SetVisibility(True)
                else:
                    self.skin_actor.SetVisibility(True)
                    self.bone_actor.SetVisibility(False)
                
        self.status.SetInput(text)
        self.ren_win.Render()

    def next(self):
        remaining_lmk = [k for k,v in self.lmk.items() if any(map(math.isnan,v))]
        for x in self.vtk_lmk:
            if x.label in remaining_lmk:
                self.style.selection = x
                self.update()
                break
        else:
            delattr(self.style, 'lmk')
            self.update('No More Landmarks!')

    def save(self):
        if not self.override or not self.file:
            self.file = asksaveasfile()
        if self.file.endswith('.csv'):
            writer = csv.writer(self.file)
            for k,v in self.lmk:
                writer.writerow([k,*v])

    def trace(self, coord, direc, option='normal'):
        # in this program, ray-tracing always happens from self.bone_actor to self.skin_actor
        # options: normal, closest, hybrid
                
        points = vtk.vtkPoints()
        cellIds = vtk.vtkIdList()
        code = self.skin_tree.IntersectWithLine(coord, [x[0]+x[1]*50 for x in zip(coord,direc)], points, cellIds)

        pointData = points.GetData()
        noPoints = pointData.GetNumberOfTuples()
        noIds = cellIds.GetNumberOfIds()

        pointsInter = []
        cellIdsInter = []
        for idx in range(noPoints):
            p = pointData.GetTuple3(idx)
            pointsInter.append(p)
            cellIdsInter.append(cellIds.GetId(idx))
            print(p, direc)
            if np.asarray(direc).dot(np.asarray(p)-np.asarray(coord))>0:
                self.lmk[self.style.selection.label] = p
                return

if __name__ == '__main__':

    d = Digitizer()
    d.setup(
        r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\n0001\20110425-pre.nii.gz',
        r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\n0001\20110425-pre-23.csv',
        )
    d.start()