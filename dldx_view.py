import os, sys, glob, csv, shutil
import numpy as np


# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingFreeType
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersSources import (
    vtkConeSource,
    vtkCubeSource,
    vtkCylinderSource,
    vtkDiskSource,
    vtkLineSource,
    vtkPlaneSource,
    vtkPointSource,
    vtkSphereSource,
    vtkTextSource
)
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkPolyDataMapper,
    vtkProperty,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTextMapper,
    vtkTextProperty
)
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtk_bridge import *
from stl import *
from kuang.digitization import landmark, library

lib = library.Library(r'kuang/cass.db')

colornames = [
            "IndianRed",
            "Pink",
            "LightSalmon",
            "Gold",
            "Lavender",
            "GreenYellow",
            "Aqua",
            "Cornsilk",
            "Gainsboro",
        ]

colors = vtkNamedColors()           

def show_case(case_name, models, landmarks, properties=None):
    
    renderer = vtkRenderer()
    renderer.SetBackground(.67, .93, .93)

    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('SourceObjectsDemo')

    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000,1500)
    renderWindow.SetWindowName(case_name)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    style = vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    for i,m in enumerate(models):
        polyd0 = vtkPolyData()
        polyd0.SetPoints(numpy_to_vtkpoints(m.v))
        polyd0.SetPolys(numpy_to_vtkpolys(m.f))

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polyd0)
        actor = vtkActor()
        actor.SetMapper(mapper)
        if properties is not None and properties[i] is not None:
            propty = actor.GetProperty()
            for pk,pv in properties[i].items():
                getattr(propty,'Set'+pk).__call__(pv)
        renderer.AddActor(actor)

    input = vtkPolyData()
    lmk = np.array(list(landmarks.values()))
    points = numpy_to_vtkpoints(lmk)
    input.SetPoints(points)

    glyphSource = vtkSphereSource()
    glyphSource.SetRadius(1)
    glyphSource.Update()

    glyph3D = vtkGlyph3D()
    glyph3D.GeneratePointIdsOn()
    glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
    glyph3D.SetInputData(input)
    glyph3D.SetScaleModeToDataScalingOff()
    glyph3D.Update()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    renderer.AddActor(actor)

    for label,coord in landmarks.items():
        txt = vtkBillboardTextActor3D()
        txt.SetPosition(*coord)
        txt.SetInput(label)
        txt.GetTextProperty().SetFontSize(24)
        txt.GetTextProperty().SetJustificationToCentered()
        txt.GetTextProperty().SetColor((0,0,0))
        txt.PickableOff()
        renderer.AddActor(txt)


    renderWindow.Render()
    interactor.Start()



def view_20230420():

    root = r'C:\data\dldx\export'
    info_sheet = r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\dldx_info.csv'
    info = {}
    
    with open(info_sheet, 'r') as f:
        info = {i+1:x for i,x in enumerate(csv.reader(f))}
    print(f'total {len(info)} case(s): ')


    sub_list = [x[0] for x in info.values()]
    box_list = glob.glob(os.path.join(root, 'DLDX*'))
    box_list = [os.path.basename(x) for x in box_list]


    for sub in sub_list:
        print(f'viewing {sub}')                    

        models_to_view = ['Skull','Mandible','Lower Teeth','Upper Teeth']
        properties = [{} for _ in range(len(models_to_view))]
        models = [() for _ in range(len(models_to_view))]

        with open(os.path.join(root, sub, 'lmk.csv'), 'r') as f:
            lmk = list(csv.reader(f))
            lmk = {x[0]:list(map(float,x[1:])) for x in lmk}

        for i,model_name in enumerate(models_to_view):
                        
            models[i] = read_stl(os.path.join(root, sub, model_name+'.stl'))
            properties[i]['Color'] = colors.GetColor3d(colornames[i])
            properties[i]['Opacity'] = 0.5
            if model_name == 'Skull':
                properties[i]['Opacity'] = 0.2

        show_case(sub, models, lmk, properties)


if __name__=='__main__':

    view_20230420()
    
