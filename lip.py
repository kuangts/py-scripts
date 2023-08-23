import os, sys, pkg_resources

import numpy as np
from vtk import *
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from tools.mesh import read_inp, write_inp, calculate_grid, remove_duplicate_nodes, boundary_faces, hex_from_numpy
from tools.polydata import polydata_from_numpy, polydata_from_points, polydata_from_plane
from tools.ui import Window, PolygonalSurfacePointSelector, MODE
from tools.image import imagedata_from_nifti


def get_surface_extent(full_extent):
    fe = full_extent

    return {
        'right':(fe[0],fe[0],fe[2],fe[3],fe[4],fe[5]),
        'left':(fe[1],fe[1],fe[2],fe[3],fe[4],fe[5]),
        'front':(fe[0],fe[1],fe[2],fe[2],fe[4],fe[5]),
        'back':(fe[0],fe[1],fe[3],fe[3],fe[4],fe[5]),
        'bottom':(fe[0],fe[1],fe[2],fe[3],fe[4],fe[4]),
        'top':(fe[0],fe[1],fe[2],fe[3],fe[5],fe[5]),
    }


def main():

    class PlanePointSelector(PolygonalSurfacePointSelector):

        def __init__(self, plane_function=None, **kwargs):
            self.plane_function = vtkPlane() if plane_function is None else plane_function
            super().__init__(**kwargs)
            self.initialize(mode=MODE.ADD)
            return None


        def new_plane(self, offset=0):
            if not hasattr(self, 'plane_index'):
                self.plane_index = 0
                p = nodes[upper_lip_ind.flat,:]/2 + nodes[lower_lip_ind.flat,:]/2
                self.lip_axial_plane_origin = p.mean(axis=0)
                p = p - self.lip_axial_plane_origin
                _,_,W = np.linalg.svd(p.T @ p) # already sorted in descending order
                self.axial_direction = W[-1,:].copy()
            new_index = self.plane_index + offset
            if new_index < 0 or new_index > lower_lip_ind.shape[0]-1:
                raise ValueError('index out of bounds')
            self.plane_index = new_index
        
            # the point in front
            p0 = nodes[upper_lip_ind[self.plane_index,0],:]/2 + nodes[lower_lip_ind[self.plane_index,0],:]/2
            # the point in the back
            p1 = nodes[upper_lip_ind[self.plane_index,-1],:]/2 + nodes[lower_lip_ind[self.plane_index,-1],:]/2
            # update the plane
            self.plane_function.SetOrigin(p0/2 + p1/2)
            plane_normal = np.cross(p0-p1, self.axial_direction)
            self.plane_function.SetNormal(plane_normal/np.sum(plane_normal**2)**.5)
            self.plane_function.Modified()
            self.render_window.Render()
            return None


        def key_press_event(self, key):
            if key == 'Left' or key == 'Right':
                offset = 1 if key == 'Left' else -1
                try:
                    origin, normal = self.new_plane(offset)
                except:
                    return None
                else:
                    self.plane_function.SetOrigin(origin)
                    self.plane_function.SetNormal(normal)
                    self.plane_function.Modified()
                    self.render_window.Render()

            return super().key_press_event(key)
    

    inp_file = r'C:\data\meshes\n0006\test\hexmesh_open.inp'
    nii_file = r'C:\data\pre-post-paired-40-send-1122\n0006\20101213-pre.nii.gz'
    w = PlanePointSelector()
    ct_lut = w.get_ct_lut()

    # read input
    img = imagedata_from_nifti(nii_file)
    nodes, elems = read_inp(inp_file)
    N = nodes.copy() # update N, and keep nodes unchanged

    # calculate node grid
    node_grid, upper_lip_ind, lower_lip_ind = calculate_grid(nodes, elems, calculate_lip_index=True)

    # adjust element order for vtkExplicitStructuredGrid - hexahedral mesh in vtk terms
    elem_order = np.lexsort([*node_grid[elems,:].sum(axis=1).T])
    elems = elems[elem_order,:]

    # construct this esgrid
    esgrid = vtkExplicitStructuredGrid()
    esgrid.SetDimensions(node_grid.max(axis=0)+1)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(nodes, deep=True))
    esgrid.SetPoints(points)
    cells = vtkCellArray()
    for c in elems:
        cells.InsertNextCell(8, c)
    esgrid.SetCells(cells)
    

    # color this esgrid
    img_interp = vtkImageInterpolator()
    img_interp.Initialize(img)
    img_interp.SetOutValue(float('nan'))
    esgrid_grayscale = numpy_to_vtk([img_interp.Interpolate(*n, 0) for n in nodes], deep=True)
    esgrid_grayscale.SetName('CT')
    esgrid.GetPointData().AddArray(esgrid_grayscale)
    grid_array = numpy_to_vtk(node_grid, deep=True)
    grid_array.SetName('Node Grid')
    esgrid.GetPointData().AddArray(grid_array)
    esgrid.GetPointData().SetActiveScalars('Node Grid')
    esgrid.ComputeFacesConnectivityFlagsArray()


    # slice the image volume with arbitrary origin and normal
    plane_cutter = vtkCutter()
    plane_cutter.SetInputData(img)
    plane_cutter.SetCutFunction(w.plane_function)
    plane_cutter.Update()
    slice_polyd = plane_cutter.GetOutput()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(plane_cutter.GetOutputPort())
    mapper.SetScalarRange(0, 4096)
    mapper.SetLookupTable(ct_lut)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOff()
    w.renderer.AddActor(actor)
    w.picker.AddPickList(actor)


    # display the esgrid outline on the slice
    converter = vtkExplicitStructuredGridToUnstructuredGrid()
    converter.SetInputData(esgrid)
    surf_filter = vtkDataSetSurfaceFilter()
    surf_filter.SetInputConnection(converter.GetOutputPort())
    surf_filter.Update()
    mesh_surf = surf_filter.GetOutput()
    contour_cut = vtkPlaneCutter()
    contour_cut.GeneratePolygonsOn()
    contour_cut.SetInputConnection(surf_filter.GetOutputPort())
    contour_cut.SetPlane(w.plane_function)
    mesh_outline = contour_cut.GetOutput()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(contour_cut.GetOutputPort())
    mapper.SetLookupTable(w.get_diverging_lut())
    mapper.SetScalarRange(0,100)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.5,.8,.8)
    w.renderer.AddActor(actor)


    # also display the entire esgrid, but hide it for now
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(surf_filter.GetOutputPort())
    mapper.SetLookupTable(w.get_diverging_lut())
    mapper.SetScalarRange(0,100)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0)
    w.renderer.AddActor(actor)
























    # # # order the points due to possible bug in contourWidget.Initialize()
    # # stripper = vtkStripper()
    # # stripper.SetInputConnection(contour_cut.GetOutputPort())
    # # stripper.JoinContiguousSegmentsOn()
    # # stripper.Update()
    # # mesh_outline = stripper.GetOutput()
    # # edges = mesh_outline.GetLines()
    # # edges = vtk_to_numpy(edges.GetConnectivityArray())
    # # points_edge = vtkPoints()
    # # lines_edge = vtkPolyLine()
    # # lines_edge.GetPointIds().SetNumberOfIds(edges.size)
    # # for i,k in enumerate(edges.flatten()):
    # #     points_edge.InsertNextPoint(mesh_outline.GetPoints().GetPoint(k))
    # #     lines_edge.GetPointIds().SetId(i,i)
    # # lines = vtkCellArray()
    # # lines.InsertNextCell(lines_edge)
    # # mesh_outline = vtkPolyData()
    # # mesh_outline.SetPoints(points_edge)
    # # mesh_outline.SetLines(lines)

    # # contourRep = vtkOrientedGlyphContourRepresentation()
    # # contourRep.GetLinesProperty().SetColor(colors.GetColor3d('Red'))
    # # contourWidget = vtkContourWidget()
    # # contourWidget.SetInteractor(w.interactor)
    # # contourWidget.SetRepresentation(contourRep)
    # # contourWidget.On()
    # # contourWidget.Initialize(mesh_outline, 1)
    # # contourWidget.Render()
    # # w.renderer.ResetCamera()
    # # w.render_window.Render()

    w.new_plane()
    w.start()


if __name__ == '__main__':
    main()