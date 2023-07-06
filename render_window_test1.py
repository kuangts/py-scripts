import numpy
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData, vtkImageData
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
from vtkmodules.vtkCommonCore import vtkDataArray, vtkScalarsToColors, vtkCharArray
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import vtkProp, vtkInteractorStyle, vtkBillboardTextActor3D
from vtk_bridge import *
from polydata_tools import *
from rendering_tools import *
from mesh_tools import *
from scipy.interpolate import RBFInterpolator





x = MeshSurfaceNodeSelector(r'C:\Users\tmhtxk25\Box\RPI\data\meshes\n0044\hexmesh_open.inp')
x.start()







