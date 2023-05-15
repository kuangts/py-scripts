# vtkRegularPolygonSource defining regions for clipping/cutting
# 2d polygon boolean
# vtkstripper
# ray cast from boundary to mandible
import glob, os, csv, shutil, re
from os.path import join as pjoin
from os.path import exists as pexists
from os.path import isfile as isfile
from os.path import isdir as isdir
from os.path import basename, dirname, normpath, realpath
from typing import Any
from vtk import vtkRegularPolygonSource, vtkPolygon

import numpy as np
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkTriangleFilter, vtkClipPolyData, vtkPolyDataConnectivityFilter, vtkImplicitPolyDataDistance
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkPolyLine, vtkUnstructuredGrid, vtkImplicitSelectionLoop, vtkPointLocator, vtkImplicitDataSet
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkFloatArray
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget, vtkBoxRepresentation
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkTransformFilter
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
    vtkRenderer,
    vtkProp3DFollower,
)
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk


from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkPolyDataNormals
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkSmoothPolyDataFilter, vtkCleanPolyData, vtkFeatureEdges, vtkStripper
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkDiscreteFlyingEdges3D, vtkTransformFilter
from vtkmodules.vtkIOImage import vtkNIFTIImageReader, vtkNIFTIImageHeader
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkCommonCore import vtkPoints

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkDataSetAttributes,
    vtkIterativeClosestPointTransform,
    vtkPlane,
    vtkBox
)
from vtkmodules.vtkFiltersCore import (
    vtkMaskFields,
    vtkThreshold,
    vtkWindowedSincPolyDataFilter
)
from vtkmodules.vtkFiltersGeneral import (
    vtkDiscreteFlyingEdges3D,
    vtkDiscreteMarchingCubes
)
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from vtkmodules.vtkImagingStatistics import vtkImageAccumulate
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkImagingMorphological import vtkImageOpenClose3D
from vtkmodules.vtkInteractionWidgets import vtkPolygonalSurfacePointPlacer
from vtkmodules.vtkFiltersModeling import vtkSelectPolyData, vtkRibbonFilter
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData
)
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
from vtk_bridge import *

colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']

colors = vtkNamedColors()

class AttrBook(object): pass


def color_disconnected_regions(input_polyd):
    fil = vtkPolyDataConnectivityFilter()
    fil.SetInputData(input_polyd)
    fil.SetExtractionModeToAllRegions()
    fil.ColorRegionsOn()
    fil.Update()
    return fil.GetOutput()


def select_connected_component(input_polyd, seed_points):


    fil = vtkPolyDataConnectivityFilter()
    fil.SetInputData(input_polyd)
    fil.SetExtractionModeToPointSeededRegions()

    locator = vtkPointLocator()
    locator.SetDataSet(input_polyd)
    locator.BuildLocator()

    for i in range(seed_points.GetNumberOfPoints()):
        id = locator.FindClosestPoint(seed_points.GetPoint(i))
        fil.AddSeed(id)

    fil.Update()

    return fil.GetOutput()

#    switch (this->GetCellType(cellId))
#    {
#      case VTK_EMPTY_CELL:
#        return 0;
#      case VTK_VERTEX:
#        return 1;
#      case VTK_LINE:
#        return 2;
#      case VTK_TRIANGLE:
#        return 3;
#      case VTK_QUAD:
#        return 4;
#      case VTK_POLY_VERTEX:
#        return this->Verts ? this->Verts->GetCellSize(this->GetCellIdRelativeToCellArray(cellId)) : 0;
#      case VTK_POLY_LINE:
#        return this->Lines ? this->Lines->GetCellSize(this->GetCellIdRelativeToCellArray(cellId)) : 0;
#      case VTK_POLYGON:
#        return this->Polys ? this->Polys->GetCellSize(this->GetCellIdRelativeToCellArray(cellId)) : 0;
#      case VTK_TRIANGLE_STRIP:
#        return this->Strips ? this->Strips->GetCellSize(this->GetCellIdRelativeToCellArray(cellId))
#                            : 0;
#    }







class ChinGuideMaker():


    def __init__(self, case_name=''):
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(.67, .93, .93)

        self.render_window = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000,1500)
        self.render_window.SetWindowName(case_name)

        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.style = vtkInteractorStyleTrackballCamera()
        self.style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(self.style)

        self.props = {}
        self.models = {}
        self.status = 'view'
        self.key_stack = ''
        self.knive_thickness = .5
        self.knive_depth = 4.
        self.picked_points = vtkPoints()
        self.picked_normals = []
        self.picked_points_actor = vtkActor()
        self.seed_points = vtkPoints()
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(.0005)
        # self.picker.InitializePickList()
        # self.picker.AddPickList(self.bone_actor)
        self.picker.SetPickFromList(False)

        # display selected points
        glyphSource = vtkSphereSource()
        glyphSource.SetRadius(1)
        glyphSource.Update()
        glyph = vtkGlyph3D()
        glyph.SetSourceConnection(glyphSource.GetOutputPort())
        pld = vtkPointSet()
        pld.SetPoints(self.picked_points)
        glyph.SetInputData(pld)
        glyph.SetScaleModeToDataScalingOff()
        glyph.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        self.picked_points_actor.SetMapper(mapper)
        self.picked_points_actor.GetProperty().SetColor(colors.GetColor3d('tomato'))
        self.renderer.AddActor(self.picked_points_actor)

        self.style.AddObserver('KeyPressEvent', self._key_pressed)
        self.style.AddObserver('CharEvent', lambda *_:None)
        self.style.AddObserver('RightButtonPressEvent', self._right_button_pressed)

        return None


    def _start(self):
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()




    def _key_pressed(self, *args):

        key = self.interactor.GetKeySym()
        ind = ''
        if key == 'Escape':
            self.key_stack = ''
        elif key == 'BackSpace' or key == 'Delete':
            self.key_stack = self.key_stack[:-1]
        elif key == 'Return':
            ind = '0'
        elif key == 'Tab':
            pass
            # print('x'*40)
            # self.key_stack = ''
        elif len(key) > 1:
            return None
        elif str.isdigit(key):
            if not self.key_stack:
                return None
            ind = key
        else:
            self.key_stack += key


        all_commands = list(filter(lambda x: callable(getattr(self, x)) and not (x.startswith('_') ) , self.__dir__()))
        all_commands_firsts = [''.join(y[0] for y in x if y) for x in [x.strip('_').split('_') for x in all_commands]]

        print('-'.join(self.key_stack+ind))
        commands = [all_commands[i] for i in [i for i,c in enumerate(all_commands_firsts) if c.startswith(self.key_stack)]]


        if not ind:
            if len(commands) == 1:
                ind = '0'
            else:
                if not commands:
                    self.key_stack = ''
                    print('x'*40)
                else:
                    print('\n'.join(f'{i}: {c}' for i,c in enumerate(commands)))
                return None
            
        try:
            cmd = commands[int(ind)]
            print('\n'.join(f'{i}: {c}' for i,c in enumerate([cmd])))
            x = getattr(self, cmd).__call__()
            if x is not None:
                print(x)
        except Exception as e:
            print(e)
        finally:
            self.key_stack = ''
            print('v'*40)

        return None



    def _right_button_pressed(self, obj, event):

        self.picker.Pick(*self.interactor.GetEventPosition(), 0, self.renderer)
        if self.status == 'view':
            obj.OnRightButtonDown()
            return None
        elif self.picker.GetCellId() == -1:
            return None
        
        coord = self.picker.GetPickPosition()
        normal = self.picker.GetPickNormal()
        self.picked_points.InsertNextPoint(*coord)
        self.picked_points.Modified()
        self.picked_normals.append(normal)

        if self.status == 'selection loop' or self.status == 'seed points' or self.status == 'incision':
            self.render_window.Render()

        if self.status == 'incision':
            self._add_to_incision(coord, normal)
                
        return None


    def _add_to_incision(self, coord, normal):
        n = self.ribbon_points.GetNumberOfPoints()//2 + 1
        normal = np.array(normal)
        coord = np.array(coord)
        self.ribbon_points.InsertNextPoint(*(coord - normal*self.knive_depth/2))
        self.ribbon_points.InsertNextPoint(*(coord + normal*self.knive_depth/2))
        self.ribbon_points.Modified()

        if n>=2:
            l0 = vtkPolyLine()
            l1 = vtkPolyLine()
            l0.GetPointIds().SetNumberOfIds(n)
            l1.GetPointIds().SetNumberOfIds(n)
            for i,k in enumerate(range(n)):
                l0.GetPointIds().SetId(i,k*2)
            for i,k in enumerate(range(n)):
                l1.GetPointIds().SetId(i,k*2+1)
            lines = vtkCellArray()
            lines.InsertNextCell(l0) 
            lines.InsertNextCell(l1) 
            self.ribbon.GetInput().SetLines(lines)
            self.ribbon.SetResolution(n*10,10)
            self.ribbon.Update()
            self.ribbon_actor.GetMapper().Update()
            self.render_window.Render()



    def _add_knive(self, point0, normal0, point1, normal1):
        

        plane_y = np.array(point1) - np.array(point0)
        d = np.sum(plane_y**2)**.5
        kniv_normal = np.cross(np.array(normal0)/2 + np.array(normal1)/2, plane_y)
        plane_x = np.cross(plane_y, kniv_normal)

        plane_x = plane_x / np.sum(plane_x**2)**.5
        plane_y = plane_y / np.sum(plane_y**2)**.5
        plane_z = np.cross(plane_x, plane_y)

        T = vtkTransform()
        T.SetMatrix(vtkMatrix4x4())
        M = share_vtkmatrix4x4_to_numpy(T.GetMatrix())
        M[:3,:3] = np.vstack((plane_x, plane_y, plane_z)).T
        M[:3,3] = np.array(point0)/2 + np.array(point1)/2
        T.GetMatrix().Modified()
        T.Scale(self.knive_depth*2, d*2, self.knive_thickness*2)

        box_rep = vtkBoxRepresentation()
        box_rep.SetTransform(T)
        polyd = vtkPolyData()
        box_rep.GetPolyData(polyd)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polyd)
        actor = vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)
        self.render_window.Render()
        
        pass

    def refresh(self):
        self.render_window.Render()

        
    def kill(self):
        yn = input('close window? (y/n) \n')
        if yn.lower() == 'y' or yn.lower() == 'yes':
            self.render_window.Finalize()
            self.render_window.End()
            del self

    
    def load_mandible(self, stl_file, **properties):

        self.load_model('0', stl_file, **properties)


    def load_model(self, name, stl_file, **properties):
        reader = vtkSTLReader()
        reader.SetFileName(stl_file)
        reader.Update()
        self.add_model(name, reader.GetOutput(), **properties)
        return None


    def add_model(self, name, polyd, **properties):

        if name in self.models or name in self.props:
            self.remove_model(name)

        cleaner = vtkCleanPolyData()
        cleaner.SetInputData(polyd)
        calc_normal = vtkPolyDataNormals()
        calc_normal.SetInputConnection(cleaner.GetOutputPort())
        calc_normal.Update()
        model = calc_normal.GetOutput()
        self.models[name] = model

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(model)
        actor = vtkActor()
        actor.SetMapper(mapper)

        if 'Color' not in properties:
            properties['Color'] = colornames.pop(0)
        if isinstance(properties['Color'], str):
            properties['Color'] = colors.GetColor3d(properties['Color'])
        for pk,pv in properties.items():
            getattr(actor.GetProperty(),'Set'+pk).__call__(pv)
        self.renderer.AddActor(actor)
        self.props[name] = actor

    
    def remove_model(self, name):
        try:
            self.renderer.RemoveActor(self.props[name])
            del self.props[name], self.models[name]
        except Exception as e:
            print(e)
        return None

    def start_view(self):
        self.status = 'view'
        self.refresh()


    def start_placing_points(self):
        self.clear_picked_points()
        self.refresh()


    def start_selection_loop(self):
        self.status = 'selection loop'
        self.start_placing_points()
                

    def start_incision(self):
        self.status = 'incision'
        self.clear_picked_points()
        self.ribbon = vtkRuledSurfaceFilter()
        self.ribbon.SetResolution(10,10)
        self.ribbon.SetRuledModeToResample()
        self.ribbon.SetOnRatio(1)
        self.ribbon.SetDistanceFactor(10000000)
        self.ribbon_points = vtkPoints()
        pldt = vtkPolyData()
        pldt.SetPoints(self.ribbon_points)
        pldt.SetLines(vtkCellArray())
        self.ribbon.SetInputData(pldt)
        self.ribbon_actor = vtkActor()
        self.ribbon_actor.SetMapper(vtkPolyDataMapper())
        self.ribbon_actor.GetMapper().SetInputConnection(self.ribbon.GetOutputPort())
        self.renderer.AddActor(self.ribbon_actor)
        self.start_placing_points()

        
    def start_seed_points(self):
        self.status = 'seed points'
        self.picked_points.DeepCopy(self.seed_points) 
        self.picked_points.Modified()
        self.refresh()  
        print(self.status)


    def quit(self):
        if self.status == 'selection loop' or self.status == 'incision':
            self.start_view()
        elif self.status == 'seed points':
            self.quit_seed_points()
        elif self.status == 'view':
            self.kill()


    def quit_placing_points(self):
        self.clear_picked_points()
        self.refresh()


    def quit_selection_loop(self):
        self.quit_placing_points()
        self.start_view()


    def quit_incision(self):
        self.quit_placing_points()
        self.renderer.RemoveActor(self.ribbon_actor)
        self.start_view()


    def quit_seed_points(self):
        self.seed_points.DeepCopy(self.picked_points)
        self.quit_placing_points()
        self.start_view()


    def clear_picked_points(self):
        self.picked_points.Reset()
        self.picked_points.Modified()
        self.picked_normals = []
        self.refresh()


    def clear_selection_loop(self):
        self.clear_picked_points()


    def clear_seed_points(self):
        self.seed_points.Reset()
        self.seed_points.Modified()
        if self.status == 'seed points':
            self.clear_picked_points()


    def cut_with_selection_loop(self):

        if self.picked_points.GetNumberOfPoints() < 3 :
            print('must have at least three points in order to cut')
            return None
            
        selector = vtkSelectPolyData()
        selector.GenerateSelectionScalarsOn()
        selector.SetEdgeSearchModeToDijkstra()
        selector.SetLoop(self.picked_points)
        selector.SetInputData(self.models['0'])
        selector.SetSelectionModeToClosestPointRegion()
        selector.SetClosestPoint(self.seed_points.GetPoint(0))
        selector.Update()

        clipper = vtkClipPolyData()
        clipper.SetInputConnection(selector.GetOutputPort())
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOn()
        clipper.Update()
        self.clear_selection_loop()

        model = clipper.GetOutput(0)
        model.GetPointData().RemoveArray(0)
        model = select_connected_component(model, self.seed_points)
        self.add_model('0', model, Color='IndianRed')

        clipped = clipper.GetOutput(1)
        clipped.GetPointData().RemoveArray(0)
        self.add_model('1', clipped, Color='Grey')
        
        self.render_window.Render()
        return None
    

    def cut_with_insicion(self):
        
        self._add_to_incision(self.picked_points.GetPoint(0), self.picked_normals[0])
        selector = vtkImplicitPolyDataDistance()
        selector.SetInput(self.ribbon.GetOutput())

        clipper = vtkClipPolyData()
        clipper.SetInputData(self.models['0'])
        clipper.SetClipFunction(selector)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOn()
        clipper.Update()

        model = clipper.GetOutput(0)
        model.GetPointData().RemoveArray(0)
        model = select_connected_component(model, self.seed_points)
        self.add_model('0', model, Color='IndianRed')

        clipped = clipper.GetOutput(1)
        clipped.GetPointData().RemoveArray(0)
        self.add_model('1', clipped, Color='Grey')
        
        self.render_window.Render()
        return None





if __name__=='__main__':

    self = ChinGuideMaker()
    self.load_mandible(r'C:\data\dldx\export\~DLDX021\Mandible.stl', Color='Silver')

    self._start()
