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
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkTriangleFilter, vtkClipPolyData, vtkPolyDataConnectivityFilter
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkPolyLine, vtkUnstructuredGrid, vtkImplicitSelectionLoop, vtkPointLocator
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList
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
from vtkmodules.vtkFiltersModeling import vtkSelectPolyData
from vtk_bridge import *

colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']

colors = vtkNamedColors()

class AttrBook(object): pass

# class namedlist(list):
#     def __init__(self, type=None):
#         super().__init__()
#         super().__setattr__('_names', ())
#         super().__setattr__('valuetype', type)

#     @property
#     def names(self):
#         return super().__getattr__('names')        

#     def __setattr__(self, __name: str, __value: Any) -> None:
#         if __name == 'valuetype':
#             print('cannot set valuetype after creation')
#         if __name == 'names':
#             print('cannot set names')
#         if __name in self.names:
#             raise ValueError(f'{__name} already exists')
#         if self.valuetype is not None and not isinstance(__value, self.valuetype):
#             raise ValueError(f'can only set value with type {self.valuetype}')
#         self.append(__value)
#         self.names += (__name,)
#         return None
    
#     def __gettattr__(self, __name: str) -> Any:
#         if __name not in self.names:
#             raise ValueError(f'{__name} does not exists')
#         return self.__getitem__(self.names.index(__name))
    
#     def __delattr__(self, __name: str) -> None:
#         if __name not in self.names:
#             raise ValueError(f'{__name} does not exists')
#         self.__delitem__(self.names.index(__name))
#         self.names.remove(__name)
#         return None
    
# def move_along_normal(polyd, amount):
#     try:
#         normals = polyd.GetPointData().GetNormals()
#         assert normals.GetNumberOfTuples() == polyd.GetPoints().GetNumberOfPoints()
#     except:
#         fil = vtkPolyDataNormals()
#         fil.SetInputData(polyd)
#         fil.Update()
#         polyd = fil.GetOutput()
#         normals = polyd.GetPointData().GetNormals()

#     normals = share_vtk_to_numpy(normals)
#     points = share_vtkpoints_to_numpy(points)
#     points_moved = points + normals * amount
#     polyd_new = vtkPolyData()
#     polyd_new.DeepCopy(polyd)
#     polyd_new.SetPoints(share_numpy_to_vtkpoints(points_moved))


#     bd_edges = vtkFeatureEdges()
#     bd_edges.ExtractAllEdgeTypesOff()
#     bd_edges.BoundaryEdgesOn()
#     bd_edges.SetInputData(polyd)
    
#     bd_strip = vtkStripper()
#     bd_strip.SetInputConnection(bd_edges.GetOutputPort())
#     bd_strip.Update()

def remove_unconnected_components(input_polyd, seed_points):

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
        renderer = vtkRenderer()
        renderer.SetBackground(.67, .93, .93)

        renderWindow = vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(1000,1500)
        renderWindow.SetWindowName(case_name)

        interactor = vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renderWindow)

        style = vtkInteractorStyleTrackballCamera()
        style.SetDefaultRenderer(renderer)
        interactor.SetInteractorStyle(style)

        self.render_window = renderWindow
        self.interactor = interactor
        self.style = style
        self.renderer = renderer
        self.props = AttrBook()
        self.models = AttrBook()
        self.status = 'view'
        self.key_stack = ''
        self.mandible_placer = vtkPolygonalSurfacePointPlacer()
        self.knive_thickness = .5
        self.knive_depth = 4
        self.placed_points = vtkPoints()
        self.seed_points = vtkPoints()
        glyphSource = vtkSphereSource()
        glyphSource.SetRadius(1)
        glyphSource.Update()

        # display selected points
        glyph = vtkGlyph3D()
        glyph.SetSourceConnection(glyphSource.GetOutputPort())
        in_points = vtkPolyData()
        in_points.SetPoints(self.placed_points)
        glyph.SetInputData(in_points)
        glyph.SetScaleModeToDataScalingOff()
        glyph.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        self.loop_actor = vtkActor()
        self.loop_actor.SetMapper(mapper)
        self.loop_actor.GetProperty().SetColor(colors.GetColor3d('tomato'))
        self.renderer.AddActor(self.loop_actor)


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
        if key == 'Escape':
            self.key_stack = ''
        elif key == 'BackSpace' or key == 'Delete':
            self.key_stack = self.key_stack[:-1]
        elif key == 'Return':
            key = '0'

        ind = ''
        if str.isdigit(key):
            if not self.key_stack:
                return
            ind = key

        elif key == 'Tab':
            print('x'*40)
            self.key_stack = ''

        elif len(key)==1:
            self.key_stack += key

        all_commands = list(filter(lambda x: callable(getattr(self, x)) and not (x.startswith('_') and not x.startswith('__')) , self.__dir__()))
        all_commands.remove('__init__')
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

        coord = [float('nan')]*3 
        orien = [float('nan')]*9
        self.mandible_placer.ComputeWorldPosition(self.renderer, self.interactor.GetEventPosition(), coord, orien)
        if self.status == 'selection loop' or self.status == 'seed points' or self.status == 'polyplane creation':
            if not np.isnan(coord).any():
                self.placed_points.InsertNextPoint(*coord)
                self.placed_points.Modified()
                self.render_window.Render()

            if self.status == 'polyplane creation':
                if self.placed_points.GetNumberOfPoints() >= 2:
                    points = share_vtkpoints_to_numpy(self.placed_points)
                    normals = share_vtk_to_numpy(self.models.mandible.GetPointData().GetNormals())
                    locator = vtkPointLocator()
                    locator.SetDataSet(self.models.mandible)
                    locator.BuildLocator()
                    point0, point1 = points[-2], points[-1]
                    normal0, normal1 = normals[locator.FindClosestPoint(point0)], normals[locator.FindClosestPoint(point1)]
                    self._add_knive(point0, normal0, point1, normal1)
        else:
            obj.OnRightButtonDown()

        return None


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

        self.load_model('mandible', stl_file, **properties)
        self.mandible_placer.AddProp(self.props.mandible)


    def load_model(self, name, stl_file, **properties):

        reader = vtkSTLReader()
        reader.SetFileName(stl_file)
        cleaner = vtkCleanPolyData()
        cleaner.SetInputConnection(reader.GetOutputPort())
        calc_normal = vtkPolyDataNormals()
        calc_normal.SetInputConnection(cleaner.GetOutputPort())
        calc_normal.Update()
        model = calc_normal.GetOutput()
        setattr(self.models, name, model)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(model)
        actor = vtkActor()
        actor.SetMapper(mapper)

        if 'Color' not in properties:
            properties['Color'] = colornames.pop(0)
        elif isinstance(properties['Color'], str):
            properties['Color'] = colors.GetColor3d(properties['Color'])
        for pk,pv in properties.items():
            getattr(actor.GetProperty(),'Set'+pk).__call__(pv)
        self.renderer.AddActor(actor)
        setattr(self.props, name, actor)

    
    def start_view(self):
        self.status = 'view'
        self.refresh()


    def start_placing_points(self):
        self.clear_placed_points()
        self.refresh()


    def start_selection_loop(self):
        self.status = 'selection loop'
        self.start_placing_points()
                

    def start_polyplane_creation(self):
        self.status = 'polyplane creation'
        self.start_placing_points()

        
    def start_seed_points(self):
        self.status = 'seed points'
        self.placed_points.DeepCopy(self.seed_points) 
        self.placed_points.Modified()
        self.refresh()  
        print(self.status)


    def quit(self):
        if self.status == 'selection loop' or self.status == 'polyplane creation':
            self.start_view()
        elif self.status == 'seed points':
            self.quit_seed_points()
        elif self.status == 'view':
            self.kill()


    def quit_placing_points(self):
        self.clear_placed_points()
        self.refresh()


    def quit_selection_loop(self):
        self.quit_placing_points()
        self.start_view()


    def quit_polyplane_creation(self):
        self.quit_placing_points()
        self.start_view()


    def quit_seed_points(self):
        self.seed_points.DeepCopy(self.placed_points)
        self.quit_placing_points()
        self.start_view()


    def clear_placed_points(self):
        self.placed_points.Reset()
        self.placed_points.Modified()
        self.refresh()


    def clear_selection_loop(self):
        self.clear_placed_points()


    def clear_seed_points(self):
        self.seed_points.Reset()
        self.seed_points.Modified()
        if self.status == 'seed points':
            self.clear_placed_points()


    def cut_with_selection_loop(self):

        if self.placed_points.GetNumberOfPoints() >=3 :
            
            selector = vtkSelectPolyData()
            selector.GenerateSelectionScalarsOn()
            selector.SetEdgeSearchModeToDijkstra()
            selector.SetLoop(self.placed_points)
            selector.SetInputData(self.models.mandible)
            selector.SetSelectionModeToClosestPointRegion()
            selector.SetClosestPoint(self.seed_points.GetPoint(0))
            selector.Update()
            # arr = share_vtk_to_numpy(selector.GetOutput().GetPointData().GetScalars())
            # arr[(arr<5)|(arr>-5)] = 0.

            clipper = vtkClipPolyData()
            clipper.SetInputConnection(selector.GetOutputPort())
            clipper.InsideOutOn()
            clipper.Update()
            self.models.mandible_cut = clipper.GetOutput()
            self.models.mandible_cut.GetPointData().RemoveArray(0)
            self.models.mandible_cut = remove_unconnected_components(self.models.mandible_cut, self.seed_points)

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(self.models.mandible_cut)
            mapper.Update()
            self.props.mandible_cut = vtkActor()
            self.props.mandible_cut.SetMapper(mapper)
            self.props.mandible_cut.GetProperty().SetColor(colors.GetColor3d('Yellow'))
            self.renderer.AddActor(self.props.mandible_cut)
            self.props.mandible.GetProperty().SetOpacity(0)
            self.mandible_placer.RemoveAllProps()
            self.mandible_placer.AddProp(self.props.mandible_cut)
            self.clear_selection_loop()
            self.render_window.Render()
        


if __name__=='__main__':

    self = ChinGuideMaker()
    self.load_mandible(r'C:\data\dldx\export\~DLDX021\Mandible.stl', Color='Silver')

    self._start()
