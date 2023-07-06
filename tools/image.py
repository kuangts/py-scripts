import numpy
from vtkmodules.vtkIOImage import vtkNIFTIImageReader, vtkNIFTIImageHeader
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtk_bridge import *


soft_tissue_threshold = (324, 1249)
bone_threshold = (1250, 4095)
foreground_threshold = (324, 4095)


def imagedata_from_nifti(file):
    src = vtkNIFTIImageReader()
    src.SetFileName(file)
    src.Update()
    matq = vtkmatrix4x4_to_numpy_(src.GetQFormMatrix())
    mats = vtkmatrix4x4_to_numpy_(src.GetSFormMatrix())
    assert numpy.allclose(matq, mats), 'nifti image qform and sform matrices not the same, requires attention'
    origin = matq[:3,:3] @ matq[:3,3]
    img = vtkImageData()
    img.DeepCopy(src.GetOutput())
    img.SetOrigin(origin.tolist())
    return img



def threshold_imagedata(image_data, thresh):
    threshold = vtkImageThreshold()
    threshold.SetInputData(image_data)
    threshold.SetInValue(1.0)
    threshold.SetOutValue(0.0)
    threshold.ReplaceInOn()
    threshold.ReplaceOutOn()
    threshold.ThresholdBetween(*thresh)
    threshold.Update()
    return threshold.GetOutput()


