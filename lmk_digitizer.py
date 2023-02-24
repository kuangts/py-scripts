import sys, os, csv, glob, math
from tkinter import Tk
Tk().withdraw()
from tkinter.filedialog import asksaveasfile
# from vtk import vtkMatrix4x4
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkCommonDataModel import vtkPointSet
from vtkmodules.vtkCommonCore import vtkPoints, reference
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
from landmark import vtkLandmark, LandmarkDict
import vtk
import numpy as np
from scipy.spatial  import KDTree
 

lmk_lib = landmark.Library().soft_tissue_23()
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
            if file.endswith('.cass'):
                lmk = LandmarkDict.from_cass(file)
            else:
                lmk = LandmarkDict.from_text(file)
        else:
            lmk = LandmarkDict()

        # keep only required landmarks
        if ordered_list is not None:
            lmk = lmk.select(ordered_list)
        print(lmk)
        return lmk

    @staticmethod
    def draw_landmark(lmk, renderer, **kwargs):
        vtk_lmk = [ vtkLandmark(k, v, **kwargs) for k,v in lmk.items() ]
        for l in vtk_lmk:
            l.refresh()
            l.set_renderer(renderer)
        return vtk_lmk

    @staticmethod
    def generate_model(source, threshold, color, **kwargs):
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
        return extractor.GetOutput(), actor

    @staticmethod
    def build_model(polyd):
        obbtree = vtk.vtkOBBTree()
        obbtree.SetDataSet(polyd)
        obbtree.BuildLocator()
        return obbtree

    @staticmethod
    def trace(point, direction, target_obbtree, option='normal'):
        # in this program, ray-tracing always happens from self.bone_actor to self.skin_actor
        # options: normal, closest, hybrid
                
        points = vtk.vtkPoints()
        cellIds = vtk.vtkIdList()
        coord = [float('nan')]*3

        if option == 'normal':
            code = target_obbtree.IntersectWithLine(point, [x[0]+x[1]*50 for x in zip(point,direction)], points, cellIds)
            pointData = points.GetData()
            intersections = [ pointData.GetTuple3(idx) for idx in range(pointData.GetNumberOfTuples()) ]
            if len(intersections):
                vec = np.asarray(intersections)-np.asarray(point)
                signed_distance = vec.dot(direction)
                ind = np.argmax(signed_distance)
                if signed_distance[ind] > 0:
                    coord = intersections[ind]

        elif option == 'closest':
            print('use closest_point instead')

        point[0],point[1],point[2] = coord[0],coord[1],coord[2]


    @staticmethod
    def closest_point(point, direction, target_kdtree, guided_by_normal=True, normal_scale_factor=1.):     
        if guided_by_normal:
            d, _ = target_kdtree.query(point, k=1)
            tr = KDTree(np.hstack((target_kdtree.data, target_kdtree.data-np.array([point]))))
            d, id = tr.query([*point, *map(lambda x:x*d, direction)], k=1)
        else:
            tr = target_kdtree
            d, id = tr.query(point, k=1)

        coord = tr.data[id]
        point[0],point[1],point[2] = coord[0],coord[1],coord[2]
        


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
        style.picker_right.SetTolerance(0.05)
        iren.SetInteractorStyle(style)

        self.ren = renderer
        self.ren_win = ren_win
        self.iren = iren
        self.style = style

        # text box for status update, soon to be replaced by PyQt
        self.status = vtkTextActor()
        # self.status.SetPosition2(10, 40)
        self.status.GetTextProperty().SetFontSize(16)
        self.status.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        self.ren.AddActor2D(self.status)

        # text box for status update, soon to be replaced by PyQt
        self.cursor = vtkTextActor()
        self.cursor.SetPosition2(4, 4)
        self.cursor.PickableOff()
        # self.cursor.GetTextProperty().SetFontSize(16)
        self.cursor.GetTextProperty().SetColor(colors.GetColor3d("Black"))
        self.ren.AddActor2D(self.cursor)

        self.ren_win.AddObserver('ModifiedEvent', self.position_panel)
        self.position_panel()

    def setup(self, nifti_file=None, ldmk_file=None):

        # clear up any existing props
        if hasattr(self, 'vtk_lmk'):
            [ a.remove() for a in self.vtk_lmk ]
        self.vtk_lmk = []
        
        # load stl models

        # first build the pipeline
        if not hasattr(self, 'reader'):

            # entry point
            self.reader = vtkNIFTIImageReader()

            # build models
            self.skin, self.skin_actor = self.generate_model(
                source=self.reader,
                threshold=(0, 324),
                color=colors.GetColor3d('peachpuff'),
                prop_name='skin'
                )
            self.bone, self.bone_actor = self.generate_model(
                source=self.reader,
                threshold=(0, 1250),
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
            self.skin_tree = self.build_model(self.skin)
            points = vtk_to_numpy(self.skin.GetPoints().GetData())
            self.kdtree = KDTree(points)

        # load landmark
        if ldmk_file is None:
            self.file = None
            self.lmk = LandmarkDict()
            self.nml = LandmarkDict()
        else:
            self.file = ldmk_file
            self.lmk = self.load_landmark(ldmk_file, get_landmark_list())
            if nifti_file is not None:
                from image import Image
                o1 = Image.read(nifti_file).GetOrigin()
                self.lmk = self.lmk - np.array(o1)

            nml = self.file.replace('.csv','-normal.csv')
            if os.path.isfile(nml):
                self.nml = self.load_landmark(nml, get_landmark_list()) # temporary
            else:
                self.nml = LandmarkDict()

        # for k,v in self.lmk.items():
        #     if k in computed:
        #         if "Zy'" in k or 'Go' in k:
        #             self.closest_point(v,self.nml[k],self.kdtree)
        #         else:
        #             self.trace(v, self.nml[k], self.skin_tree)

        if len(self.lmk):

            # landmark props self.vtk_lmk
            vtk_computed = self.draw_landmark(LandmarkDict(zip(computed, self.lmk.coordinates(computed))), self.ren, Color=(1,0,1), HighlightColor=(1,1,0))
            vtk_digitized = self.draw_landmark(LandmarkDict(zip(digitized,self.lmk.coordinates(digitized))), self.ren)
            self.vtk_lmk = vtk_computed + vtk_digitized

            # select the first landmark to start
            self.style.selection = self.vtk_lmk[0]


            self.update()

    def start(self):
        self.iren.Start()

    def position_panel(self, obj=None, event=None):
        s = self.ren_win.GetSize()
        s0 = [float('nan')]*2
        self.status.GetSize(self.ren, s0)
        # self.status.SetPosition(s[0]*.8,s[1]*.9-s0[1])
        self.status.SetPosition(0,0)


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

    def right_button_press_event(self, obj, event):
        obj.pick_mode = True
        obj.right_button = True

    def mouse_move(self, obj, event):
        obj.pick_mode = False
        pos = self.iren.GetEventPosition()
        # if not (hasattr(obj, 'right_button') and obj.right_button or hasattr(obj, 'left_button') and obj.left_button):
        #     obj.picker_right.Pick(pos[0], pos[1], 0, self.ren)
        #     self.cursor.SetPosition(pos[0], pos[1])
        #     coord_current = self.lmk[obj.selection.label]
        #     direc_current = self.nml[obj.selection.label]
        #     if not np.isnan(coord_current).any() and obj.picker_right.GetCellId() != -1:
        #         coord = np.asarray((obj.picker_right.GetPickPosition()))
        #         direc = np.asarray((obj.picker_right.GetPickNormal()))
        #         vec = coord - coord_current
        #         d = np.sum((coord - coord_current)**2)**.5
        #         a = np.arccos(vec.dot(direc_current)/np.sum(vec**2)**.5)/np.pi*180
        #         self.cursor.SetInput(f'dist:{d:.2f}\nangle:{a:.1f}')
        #         self.ren_win.Render()
        obj.OnMouseMove()

    def left_button_release_event(self, obj, event):
        if obj.pick_mode and obj.left_button:
            pos = self.iren.GetEventPosition()
            obj.picker_left.Pick(pos[0], pos[1], 0, self.ren)
            if obj.picker_left.GetCellId() != -1:
                prop = obj.picker_left.GetProp3D()
                # obj.HighlightProp3D(prop)
                if prop.prop_name == 'ldmk':
                    if hasattr(obj,'selection'):
                        obj.selection.deselect()
                    obj.selection = prop.parent
                    obj.selection.select()
                self.update()
        # Forward events
        obj.pick_mode = True
        obj.left_button = False
        obj.OnLeftButtonUp()

    def right_button_release_event(self, obj, event):
        if obj.pick_mode and obj.right_button:
            pos = self.iren.GetEventPosition()
            obj.picker_right.Pick(pos[0], pos[1], 0, self.ren)
            if obj.picker_right.GetCellId() != -1:
                coord = list(obj.picker_right.GetPickPosition())
                direc = list(obj.picker_right.GetPickNormal())
                if self.iren.GetControlKey() and hasattr(self, 'skin_tree'):
                    if "Zy'" in obj.selection.label or "Go'" in obj.selection.label:
                        self.closest_point(coord, direc, self.kdtree)
                    else:
                        self.trace(coord, direc, self.skin_tree)
                self.lmk[obj.selection.label] = coord
                obj.selection.move_to(coord)
                self.update()
        # Forward events
        obj.pick_mode = True
        obj.right_button = False

    def update(self, text=''):
        if not text and hasattr(self.style, 'selection'):
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
            # delattr(self.style, 'selection')
            self.update('No More Landmarks!')

    def save(self):
        if not self.override or not self.file:
            self.file = asksaveasfile()
        if self.file.endswith('.csv'):
            writer = csv.writer(self.file)
            for k,v in self.lmk:
                writer.writerow([k,*v])


if __name__ == '__main__':

    d = Digitizer()
    d.setup(
        r'C:\data\pre-post-paired-40-send-1122\n0002\20100921-pre.nii.gz',
        r'C:\data\pre-post-paired-soft-tissue-lmk-23\n0002\skin-pre-23.csv',
        )
    d.start()