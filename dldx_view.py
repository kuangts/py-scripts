import os, sys, glob, csv
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
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtk_bridge import *
from stl import *


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
                

def show_case(models, properties=None):
    
    renderer = vtkRenderer()
    renderer.SetBackground(.67, .93, .93)

    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('SourceObjectsDemo')

    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000,1500)

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

    renderWindow.Render()
    interactor.Start()



def view(root):

    info_sheet = os.path.join(root, 'info.csv')
    info = {}
    
    with open(info_sheet, 'r') as f:
        info = {i+1:x for i,x in enumerate(csv.reader(f))}
    print(f'total {len(info)} case(s): ')

    sub_list = [x[0] for x in info.values()]
    for sub in sub_list:
        print(f'viewing {sub}')                    

        models_to_view = ['CT Soft Tissue','Skull','Mandible','Lower Teeth','Upper Teeth']
        properties = [{} for _ in range(len(models_to_view))]
        models = [() for _ in range(len(models_to_view))]

        for i,m in enumerate(models_to_view):
                        
            models[i] = read_stl(os.path.join(root, sub, m+'.stl'))
            properties[i]['Color'] = vtkNamedColors().GetColor3d(colornames[i])
            if m == 'CT Soft Tissue':
                properties[i]['Opacity'] = .2

        show_models(models, properties)


if __name__=='__main__':

    # root = os.path.normpath(sys.argv[1])
    root = os.path.normpath(r'C:\data\dldx\export')
    view(root)
    
