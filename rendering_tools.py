import numpy
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkImageData
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
    vtkColorTransferFunction,
    
)
from vtkmodules.vtkCommonCore import vtkDataArray, vtkScalarsToColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
from vtk_bridge import *

colornames = ['IndianRed', 'LightSalmon', 'Pink', 'Gold', 'Lavender', 'GreenYellow', 'Aqua', 'Cornsilk', 'White', 'Gainsboro',
              'LightCoral', 'Coral', 'LightPink', 'Yellow', 'Thistle', 'Chartreuse', 'Cyan', 'BlanchedAlmond', 'Snow', 'LightGrey',
              'Salmon', 'Tomato', 'HotPink', 'LightYellow', 'Plum', 'LawnGreen', 'LightCyan', 'Bisque', 'Honeydew','Silver',
              'DarkSalmon', 'OrangeRed', 'DeepPink', 'LemonChiffon', 'Violet', 'Lime', 'PaleTurquoise', 'NavajoWhite', 'MintCream',
              'DarkGray', 'LightSalmon', 'DarkOrange', 'MediumVioletRed', 'LightGoldenrodYellow', 'Orchid', 'LimeGreen', 'Aquamarine', 'Wheat', 'Azure', 'Gray',
              'Red', 'Orange', 'PaleVioletRed', 'PapayaWhip', 'Fuchsia', 'PaleGreen', 'Turquoise', 'BurlyWood', 'AliceBlue', 'DimGray', 'Crimson']

colors = vtkNamedColors()

def build_color(color_range, symmetric_map=False):
    lut = vtkColorTransferFunction()
    r = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5]
    g = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    b = [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rgb = numpy.vstack((r,g,b),dtype=numpy.float64).T
    if symmetric_map:
        rgb = numpy.vstack((rgb[:0:-1,:], rgb))
    lut.BuildFunctionFromTable(*color_range, rgb.shape[0], rgb.ravel())
    lut.Modified()
    return lut

def map_scalars_through_table(lut:vtkScalarsToColors, scalars):
    if isinstance(scalars, numpy.ndarray):
        sc = scalars.flatten()
        rgba = numpy.empty(sc.size*4, "uint8")
        sc_vtk = numpy_to_vtk_(sc)
        shp = scalars.shape
    else:
        sc_vtk = scalars
        shp = (scalars.GetNumberOfTuples(),)
    rgba = numpy.empty(sc_vtk.GetNumberOfTuples()*4, "uint8")
    lut.MapScalarsThroughTable(sc_vtk, rgba)
    rgba = rgba.reshape(*shp,4)/255 #???????
    return rgba


def render_window(window_title):
    renderer = vtkRenderer()
    renderer.SetBackground(.67, .93, .93)

    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000,1500)
    renderWindow.SetWindowName(window_title)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    style = vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)
    return renderWindow, renderer, interactor, style


def polydata_actor(polyd:vtkPolyData, **property):
    
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polyd)
    actor = vtkActor()
    actor.SetMapper(mapper)
    if property:
        for pk,pv in property.items():
            if pk=='Color' and isinstance(pv, str):
                pv = colors.GetColor3d(pv)
            getattr(actor.GetProperty(),'Set'+pk).__call__(pv)
    return actor


def text_actor(coords:numpy.ndarray, label:str, font_size=24, color=(0,0,0), display_offset=(0,10), **text_property):
    if isinstance(color, str):
        color = colors.GetColor3d(color)

    actor = vtkBillboardTextActor3D()
    actor.SetPosition(coords)
    actor.SetInput(label)
    actor.SetDisplayOffset(*display_offset)
    actor.GetTextProperty().SetFontSize(font_size)
    actor.GetTextProperty().SetColor(color)
    actor.GetTextProperty().SetJustificationToCentered()
    actor.PickableOff()
    # actor.ForceOpaqueOn()
    if text_property:
        for pk,pv in text_property.items():
            getattr(actor.GetTextProperty(),'Set'+pk).__call__(pv)


def test():
    lut = build_color([-6,6], symmetric_map=True)
    s = map_scalars_through_table(lut, numpy.vstack((numpy.arange(7),-numpy.arange(7))))
    print(s)


if __name__=='__main__':
    test()


