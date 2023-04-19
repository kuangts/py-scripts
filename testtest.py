#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from collections import namedtuple
import numpy as np
import struct
from vtk_bridge import *

from CASS import CASS

def read_stl(files):
    models = [None]*len(files)
    Model = namedtuple('Model',('v','f','fn'))
    for i,file in enumerate(files):
        with open(file, 'br') as f:
            f.seek(80)
            data = f.read()
        nf, data = struct.unpack('I', data[0:4])[0], data[4:]
        data = struct.unpack('f'*(nf*12), b''.join([data[i*50:i*50+48] for i in range(nf)]))
        data = np.asarray(data).reshape(-1,12)
        FN = data[:,0:3].astype(np.float32)
        V = data[:,3:12].reshape(-1,3).astype(np.float32)
        F = np.arange(0,len(V)).reshape(-1,3).astype(np.int64)
        models[i] = Model(v=V, f=F, fn=FN)

    return models





def show_models(cass_file=r'C:\data\dldx\kuang\Alasa^Esther^KG.CASS'):

    with CASS(cass_file) as f:
        models = f.load_models(['Mandible'])

    models_correct = read_stl([r'C:\data\dldx\kuang\Alasa^Esther\Mandible Correct.stl'])

    renderer = vtkRenderer()
    renderer.SetBackground(.67, .93, .93)

    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('SourceObjectsDemo')

    m = models[0]
    polyd0 = vtkPolyData()
    polyd0.SetPoints(numpy_to_vtkpoints(m.v))
    polyd0.SetPolys(numpy_to_vtkpolys(m.f))
    polyd0t = polyd0

    mtx = vtkMatrix4x4()
    arr = np.array(m.T[0])
    mtx.DeepCopy(arr.ravel())
    fil = vtkMatrixToLinearTransform()
    fil.SetInput(mtx)
    ffil = vtkTransformPolyDataFilter()
    ffil.SetTransform(fil)
    ffil.SetInputData(polyd0)
    ffil.Update()
    polyd0t = ffil.GetOutput()


    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polyd0t)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.0,0.5,0.5)
    renderer.AddActor(actor)

    m = models_correct[0]
    polyd1 = vtkPolyData()
    polyd1.SetPoints(numpy_to_vtkpoints(m.v))
    polyd1.SetPolys(numpy_to_vtkpolys(m.f))

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polyd1)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.5,0.5,0.5)
    renderer.AddActor(actor)

    renderWindow.AddRenderer(renderer)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    style = vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    renderWindow.Render()
    interactor.Start()


if __name__ == '__main__':
    show_models()