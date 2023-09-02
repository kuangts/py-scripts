import numpy as np
import SimpleITK as sitk
from vtkmodules.vtkIOImage import vtkNIFTIImageReader, vtkNIFTIImageHeader
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtk_bridge import *


soft_tissue_threshold = (324, 1249)
bone_threshold = (1250, 4095)
foreground_threshold = (324, 4095)


def write_imagedata_to_nifti(image_data, file):
    sitk.WriteImage(imagedata_to_sitk(image_data), file)


def imagedata_from_nifti(file):
    return imagedata_from_sitk(sitk.ReadImage(file))


def imagedata_from_sitk(sitk_img):
    voxels = sitk.GetArrayFromImage(sitk_img)
    voxels = numpy_to_vtk(voxels.flatten(), deep=True)
    vtk_img = vtkImageData()
    vtk_img.GetPointData().SetScalars(voxels)
    vtk_img.SetOrigin(sitk_img.GetOrigin())
    vtk_img.SetDimensions(sitk_img.GetSize())
    vtk_img.SetSpacing(sitk_img.GetSpacing())
    vtk_img.SetDirectionMatrix(sitk_img.GetDirection())
    return vtk_img


def imagedata_to_sitk(vtk_img):
    arr = vtk_to_numpy(vtk_img.GetPointData().GetScalars()).reshape(vtk_img.GetDimensions()[::-1])
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetOrigin(vtk_img.GetOrigin())
    sitk_img.SetSpacing(vtk_img.GetSpacing())
    X = np.array([vtk_img.GetDirectionMatrix().GetElement(*ij) for ij in np.ndindex((3,3))])
    sitk_img.SetDirection(X.tolist())
    return sitk_img


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


