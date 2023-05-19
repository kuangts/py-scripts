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
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkParametricFunctionSource
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkTriangleFilter, vtkClipPolyData, vtkPolyDataConnectivityFilter, vtkImplicitPolyDataDistance, vtkAppendPolyData
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkPolyLine, vtkUnstructuredGrid, vtkImplicitSelectionLoop, vtkPointLocator, vtkImplicitDataSet
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkPoints, reference, vtkPoints, vtkIdList, vtkFloatArray
from vtkmodules.vtkInteractionWidgets import vtkPointCloudRepresentation, vtkPointCloudWidget, vtkBoxRepresentation, vtkContourWidget
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkTransformFilter, vtkBooleanOperationPolyDataFilter 
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
    vtkCoordinate
)
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from vtkmodules.vtkCommonComputationalGeometry import vtkParametricSpline
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
from vtkmodules.vtkInteractionWidgets import vtkPolygonalSurfacePointPlacer, vtkOrientedGlyphContourRepresentation
from vtkmodules.vtkFiltersModeling import vtkSelectPolyData, vtkRibbonFilter
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData
)
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
from vtkmodules.vtkInteractionWidgets import vtkWidgetEvent




from vtk_bridge import *

colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']

colors = vtkNamedColors()


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


def color_disconnected_regions(input_polyd:vtkPolyData):
    fil = vtkPolyDataConnectivityFilter()
    fil.SetInputData(input_polyd)
    fil.SetExtractionModeToAllRegions()
    fil.ColorRegionsOn()
    fil.Update()
    return fil.GetOutput()


def select_connected_component(input_polyd:vtkPolyData, seed_points:vtkPoints):

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


def extrude_surface(polyd:vtkPolyData, extrude_length=1):
    normals = share_vtk_to_numpy(polyd.GetPointData().GetNormals())
    points = share_vtkpoints_to_numpy(polyd.GetPoints())
    new_points = points + normals * extrude_length
    outer = vtkPolyData()
    outer.DeepCopy(polyd)
    outer.SetPoints(share_numpy_to_vtkpoints(new_points))

    edge_filter = vtkFeatureEdges()
    edge_filter.ExtractAllEdgeTypesOn()
    edge_filter.BoundaryEdgesOn()
    edge_filter.SetInputData(polyd)
    stripper = vtkStripper()
    stripper.SetInputConnection(edge_filter.GetOutputPort())
    stripper.Update()
    stripper_points = share_vtkpoints_to_numpy(stripper.GetOutput().GetPoints())
    stripper_normals = share_vtk_to_numpy(stripper.GetOutput().GetPointData().GetNormals())
    new_stripper_points = stripper_points + stripper_normals * extrude_length
    edges = stripper.GetOutput().GetLines()
    edges = share_vtkpolys_to_numpy(edges)
    new_edges = edges + stripper_points.shape[0]
    all_points = np.vstack((stripper_points, new_stripper_points))
    all_points = share_numpy_to_vtkpoints(all_points)

    ribbon = vtkRuledSurfaceFilter()
    ribbon.SetRuledModeToPointWalk()
    ribbon.SetDistanceFactor(100000)
    ribbon_polyd = vtkPolyData()
    l0 = vtkPolyLine()
    l1 = vtkPolyLine()
    l0.GetPointIds().SetNumberOfIds(edges.size)
    l1.GetPointIds().SetNumberOfIds(edges.size)
    for i,k in enumerate(edges.flatten()):
        l0.GetPointIds().SetId(i,k)
    for i,k in enumerate(new_edges.flatten()):
        l1.GetPointIds().SetId(i,k)
    lines = vtkCellArray()
    lines.InsertNextCell(l0)
    lines.InsertNextCell(l1)
    ribbon_polyd.SetLines(lines)
    ribbon_polyd.SetPoints(all_points)
    ribbon.SetInputData(ribbon_polyd)
    ribbon.Update()
    skirt = ribbon.GetOutput()
    
    append = vtkAppendPolyData()
    append.AddInputData(polyd)
    append.AddInputData(outer)
    append.AddInputData(skirt)
    cleaner = vtkCleanPolyData()
    cleaner.SetInputConnection(append.GetOutputPort())
    cleaner.Update()

    return cleaner.GetOutput()



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
        self.knife_depth = 12.
        self.extrude_length = 2.
        self.picked_points = vtkPoints()
        self.picked_normals = []
        self.seed_points = vtkPoints()
        self.picker = vtkCellPicker()
        self.display_coordinate = vtkCoordinate()
        self.display_coordinate.SetCoordinateSystemToDisplay()
        self.display_coordinate.SetViewport(self.renderer)
        
        self.picker.SetTolerance(.0005)
        # self.picker.InitializePickList()
        # self.picker.AddPickList(self.bone_actor)
        self.picker.SetPickFromList(False)

        # display picked points
        self.add_points('picked_points', self.picked_points, Color='tomato')

        # display seed points
        self.add_points('seed_points', self.seed_points, radius=.5, Color='Yellow')

        self.style.AddObserver('KeyPressEvent', self._key_pressed)
        self.style.AddObserver('CharEvent', lambda *_:None)
        self.style.AddObserver('RightButtonPressEvent', self._right_button_pressed)

        return None


    def refresh(self):
        self.render_window.Render()

        
    def kill(self):
        yn = input('close window? (y/n) \n')
        if yn.lower() == 'y' or yn.lower() == 'yes':
            self.render_window.Finalize()
            self.render_window.End()
            del self


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

        # 'view' is the default style
        if self.status == 'view':
            obj.OnRightButtonDown()
            return None
        
        # some actions do not require a hit on object
        # some actions demand separate handling

        if self.status == 'slab intersect':
            self.display_coordinate.SetValue(*self.interactor.GetEventPosition(), 0)
            world_coord = self.display_coordinate.GetComputedWorldValue(self.renderer)
            view_dir = self.renderer.GetActiveCamera().GetDirectionOfProjection()
            # print(world_coord)
            # print(view_dir)
            # self.picked_points.InsertNextPoint(*world_coord)
            # self.picked_points.Modified()
            return None
            
            
        # some actions require a hit on object
        # all action statuses (all except 'view') use populated picked_points
        self.picker.Pick(*self.interactor.GetEventPosition(), 0, self.renderer)
        if self.picker.GetCellId() != -1:
            coord = self.picker.GetPickPosition()
            normal = self.picker.GetPickNormal()

            self.picked_points.InsertNextPoint(*coord)
            self.picked_points.Modified()
            self.picked_normals.append(normal)

            if self.status == 'seed points':
                self._add_seed_point(coord)
            elif self.status == 'ribbon clip':
                self._add_ribbon_points(coord, normal)

        # some actions require a hit in negative space
        else:
            pass


        self.refresh()
                
        return None


    def _right_botton_moved(self, obj, event):
        # some actions do not require a hit on object
        if self.status == 'slab intersect':
            coordinate = vtkCoordinate()
            coordinate.SetCoordinateSystemToDisplay()
            coordinate.SetValue(*self.interactor.GetEventPosition(), 0)
            world_coord = coordinate.GetComputedWorldValue(self.renderer)


    def _add_seed_point(self, coord):
        self.seed_points.InsertNextPoint(coord)
        self.seed_points.Modified()
        self.props['seed_points'].GetMapper().Update()


    def _add_ribbon_points(self, coord, normal):
        normal = np.array(normal)
        coord = np.array(coord)
        self.ribbon_points.InsertNextPoint(*(coord - normal*self.knife_depth/2))
        self.ribbon_points.InsertNextPoint(*(coord + normal*self.knife_depth/2))
        self.ribbon_points.Modified()
        n = self.ribbon_points.GetNumberOfPoints()//2

        if n>1:
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
        
        return None


    def _add_contour_points(self, coord):
        normal = np.array(normal)
        coord = np.array(coord)
        self.ribbon_points.InsertNextPoint(*(coord - normal*self.knife_depth/2))
        self.ribbon_points.InsertNextPoint(*(coord + normal*self.knife_depth/2))
        self.ribbon_points.Modified()
        n = self.ribbon_points.GetNumberOfPoints()//2

        if n>1:
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
        
        return None


    def _add_slab(self, point0, normal0, point1, normal1):
    
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
        T.Scale(self.knife_depth*2, d*2, self.knive_thickness*2)

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

    
    def load_mandible(self, stl_file, **properties):

        self.load_model('0', stl_file, **properties)


    def load_model(self, name, stl_file, **properties):
        reader = vtkSTLReader()
        reader.SetFileName(stl_file)
        reader.Update()
        self.add_model(name, reader.GetOutput(), **properties)
        return None


    def add_model(self, name:str, polyd:vtkPolyData, **properties):

        if name in self.models or name in self.props:
            self.remove_model(name)

        cleaner = vtkCleanPolyData()
        cleaner.SetInputData(polyd)
        calc_normal = vtkPolyDataNormals()
        calc_normal.SetInputConnection(cleaner.GetOutputPort())
        calc_normal.Update()
        self.models[name] = calc_normal.GetOutput()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(calc_normal.GetOutputPort())
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


    def add_points(self, name:str, points:vtkPoints, radius=1, **properties):
        
        glyphSource = vtkSphereSource()
        glyphSource.SetRadius(radius)
        glyphSource.Update()
        glyph = vtkGlyph3D()
        glyph.SetSourceConnection(glyphSource.GetOutputPort())
        pld = vtkPointSet()
        pld.SetPoints(points)
        glyph.SetInputData(pld)
        glyph.SetScaleModeToDataScalingOff()
        glyph.Update()
        self.models[name] = glyph.GetOutput()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
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
        return None

    
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


    def start_picking_points(self):
        self.clear_picked_points()
        self.refresh()


    def start_seed_points(self):
        self.status = 'seed points'
        self.refresh()


    def start_selection_loop(self):
        self.status = 'selection loop'
        self.start_picking_points()
                

    def start_contour_clip(self):
        
        self.status = 'contour clip'
        self.contour_widget = vtkContourWidget()
        self.contour_widget.SetInteractor(self.interactor)
        self.contour_widget.AddObserver(vtkWidgetEvent.GetEventIdFromString('Select'), self._contour_widget_select)
        self.contour_widget.AddObserver(vtkWidgetEvent.GetEventIdFromString('AddFinalPoint'), self._contour_widget_add_final_point)
        pointPlacer = vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(self.props['0'])

        rep = self.contour_widget.GetRepresentation()
        rep.GetLinesProperty().SetColor(colors.GetColor3d("Crimson"))
        rep.GetLinesProperty().SetLineWidth(3.0)
        rep.SetPointPlacer(pointPlacer)
        # rep.ClosedLoopOn() # crush if set

        self.contour_widget.EnabledOn()

        self.ribbon = vtkRuledSurfaceFilter()
        self.ribbon.SetRuledModeToPointWalk()
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
        return None


    def _contour_widget_select(self, obj, event):
        rep = self.contour_widget.GetRepresentation()
        polyd = rep.GetContourRepresentationAsPolyData()
        if not polyd.GetPoints() or not polyd.GetPoints().GetNumberOfPoints():
            return
        coords = share_vtkpoints_to_numpy(polyd.GetPoints())
        locator = vtkPointLocator()
        locator.SetDataSet(self.models['0'])
        locator.BuildLocator()
        ids = []
        for i in range(polyd.GetPoints().GetNumberOfPoints()):
            ids.append(locator.FindClosestPoint(coords[i].tolist()))
        normals = share_vtk_to_numpy(self.models['0'].GetPointData().GetNormals())
        normals = normals[ids]

        self.ribbon_points.DeepCopy(share_numpy_to_vtkpoints(np.vstack((coords-normals,coords+normals))))
        self.ribbon_points.Modified()
        n = self.ribbon_points.GetNumberOfPoints()//2

        if n>1:
            l0 = vtkPolyLine()
            l1 = vtkPolyLine()
            l0.GetPointIds().SetNumberOfIds(n)
            l1.GetPointIds().SetNumberOfIds(n)
            for i,k in enumerate(range(n)):
                l0.GetPointIds().SetId(i,k)
                l1.GetPointIds().SetId(i,k+n)
            lines = vtkCellArray()
            lines.InsertNextCell(l0) 
            lines.InsertNextCell(l1) 
            self.ribbon.GetInput().SetLines(lines)
            self.ribbon.Update()
            self.ribbon_actor.GetMapper().Update()
        
        return None



    def _contour_widget_add_final_point(self, obj, event):
        pass



    def start_ribbon_clip(self):
        self.status = 'ribbon clip'
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
        self.start_picking_points()
        return None


    def start_slab_clip(self):
        self.status = 'slab intersect'
        self.clear_picked_points()

        # if self.picked_points.GetNumberOfPoints() == 1:
        #     return
        
        # elif self.picked_points.GetNumberOfPoints() == 2:
        #     self.ribbon = vtkRuledSurfaceFilter()
        #     # self.ribbon.SetResolution(10,10)
        #     self.ribbon.SetRuledModeToPointWalk()
        #     self.ribbon.SetOnRatio(1)
        #     self.ribbon.SetDistanceFactor(10000000)
        #     self.ribbon_points = vtkPoints()
        #     pldt = vtkPolyData()
        #     pldt.SetPoints(self.ribbon_points)
        #     pldt.SetLines(vtkCellArray())
        #     self.ribbon.SetInputData(pldt)
        #     self.ribbon_actor = vtkActor()
        #     self.ribbon_actor.SetMapper(vtkPolyDataMapper())
        #     self.ribbon_actor.GetMapper().SetInputConnection(self.ribbon.GetOutputPort())
        #     self.renderer.AddActor(self.ribbon_actor)
        #     self.start_picking_points()

        # else:
        #     raise ValueError("shouldn't be here")

        return None


    def quit_picking_points(self):
        self.clear_picked_points()
        self.refresh()


    def quit_selection_loop(self):
        self.quit_picking_points()
        self.start_view()


    def quit_ribbon_clip(self):
        self.quit_picking_points()
        self.renderer.RemoveActor(self.ribbon_actor)
        del self.ribbon_points, self.ribbon, self.ribbon_actor
        self.start_view()


    def quit_contour_clip(self):
        self.contour_widget.EnabledOff()
        del self.contour_widget
        self.start_view()


    def quit_seed_points(self):
        self.quit_picking_points()
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


    def clip_with_selection_loop(self):

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
        
        self.refresh()
        return None
    

    def close_ribbon_and_clip(self):
        
        self._add_ribbon_points(self.picked_points.GetPoint(0), self.picked_normals[0])
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
        
        self.refresh()
        return None



    def ribbon_clip(self):
        
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
        
        self.refresh()
        return None




    def close_ribbon_and_boolean(self):
        
        self._add_ribbon_points(self.picked_points.GetPoint(0), self.picked_normals[0])
        diff_filter = vtkBooleanOperationPolyDataFilter()
        diff_filter.SetOperationToDifference()
        diff_filter.SetInputData(0, self.models['0'])
        diff_filter.SetInputData(1, self.ribbon.GetOutput())
        diff_filter.Update()

        model = diff_filter.GetOutput(0)
        model.GetPointData().RemoveArray(0)
        model = select_connected_component(model, self.seed_points)
        self.add_model('0', model, Color='IndianRed')

        # clipped = diff_filter.GetOutput(1)
        # clipped.GetPointData().RemoveArray(0)
        # self.add_model('1', clipped, Color='Grey')
        
        self.refresh()
        return None


    def extrude(self):
        polyd = self.models['0']
        model = extrude_surface(polyd, self.extrude_length)
        self.add_model('0', model, Color='IndianRed')
        self.refresh()
        



if __name__=='__main__':

    self = ChinGuideMaker()
    self.load_mandible(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\manu-mand.stl', Color='Silver')
    # self._add_seed_point((78.32646120703109, 12.064000941478483, 24.00454019563039))
    self._start()
