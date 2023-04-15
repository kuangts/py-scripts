import os, sys, glob
import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
colors = vtkNamedColors()
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkSmoothPolyDataFilter
from vtkmodules.vtkCommonTransforms import vtkMatrixToLinearTransform, vtkLinearTransform, vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkDiscreteFlyingEdges3D, vtkTransformFilter
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
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
    vtkDataSetAttributes
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


import image

def main(sub_dir, nifti_file_pre=None, new_dicom_dir_pre=None, seg_post=None, stl_file_post=None, transform_file=None, ):

    os.chdir(sub_dir)
    sub = os.path.basename(sub_dir)

    if nifti_file_pre is None:
        nifti_file_pre = glob.glob('*-pre.nii.gz')[0]

    if new_dicom_dir_pre is None:
        new_dicom_dir_pre = nifti_file_pre.replace('.nii.gz', '')
        
    img = image.Image.read(nifti_file_pre)
    img = image.Image(img - 1024)
    img.SetOrigin((0,0,0))
    img.write_gdcm(new_dicom_dir_pre)

    if seg_post is None:
        seg_post = glob.glob('*-post-seg.nii.gz')[0]

    if transform_file is None:
        transform_file = sub+'_reset_origin.tfm'
    
    if stl_file_post is None:
        stl_file_post = 'bone_post.stl'

    nifti_to_stl(seg_post, stl_file_post, transform=transform_file)

def nifti_to_stl(mask_file, 
                 background_value=0.0,
                 foreground_value_tuple=(1.0,2.0),
                 union=False, # if True, create one 
                 smooth_on=True,
                 smooth_parameter=None,
                 transform=None):

    # Generate models from labels
    # 1) Read the meta file
    # 2) Generate a histogram of the labels
    # 3) Generate models from the labeled volume
    # 4) Smooth the models
    # 5) Output each model into a separate file


    reader = vtkNIFTIImageReader()
    threshold = vtkImageThreshold()
    closer = vtkImageOpenClose3D()
    discrete_cubes = vtkDiscreteFlyingEdges3D()
    smoother = vtkWindowedSincPolyDataFilter()
    scalars_off = vtkMaskFields()
    geometry = vtkGeometryFilter()
    writer = vtkSTLWriter()

    smoothing_iterations = 15
    pass_band = 0.001
    feature_angle = 120.0

    reader.SetFileName(mask_file)

    port = reader.GetOutputPort()
    
    threshold.SetInputConnection(port)
    threshold.SetInValue(1.0)
    threshold.SetOutValue(0.0)
    threshold.ReplaceInOn()
    threshold.ReplaceOutOn()
    threshold.ThresholdByUpper(0.5)

    closer.SetInputConnection(threshold.GetOutputPort())
    closer.SetKernelSize(2, 2, 2)
    closer.SetCloseValue(1.0)

    discrete_cubes.SetInputConnection(closer.GetOutputPort())
    discrete_cubes.SetValue(0, 1.0)
    
    scalars_off.SetInputConnection(discrete_cubes.GetOutputPort())
    scalars_off.CopyAttributeOff(vtkMaskFields().POINT_DATA,
                                 vtkDataSetAttributes().SCALARS)
    scalars_off.CopyAttributeOff(vtkMaskFields().CELL_DATA,
                                 vtkDataSetAttributes().SCALARS)

    geometry.SetInputConnection(scalars_off.GetOutputPort())

    smoother.SetInputConnection(geometry.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()

    writer_port = smoother.GetOutputPort()
    
    if transform_file is not None:
        T = np.genfromtxt(transform_file).ravel()
        M = vtkMatrix4x4()
        M.DeepCopy(T)
        T = vtkMatrixToLinearTransform()
        T.SetInput(M)
        transform = vtkTransformFilter()
        transform.SetTransform(T)
        transform.SetInputConnection(smoother.GetOutputPort())
        writer_port = transform.GetOutputPort()

    writer = vtkSTLWriter()
    writer.SetInputConnection(writer_port)
    writer.SetFileName(stl_file)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()    


def transform_stl(input, output, T):
    reader = vtkSTLReader()
    reader.SetFileName(input)
    transform = vtkMatrixToLinearTransform()
    transform.SetInput(vtkMatrix4x4())
    transform.GetInput().DeepCopy(np.genfromtxt(T).ravel())
    register = vtkTransformPolyDataFilter()
    register.SetTransform(transform)
    register.SetInputConnection(reader.GetOutputPort())
    writer = vtkSTLWriter()
    writer.SetInputConnection(register.GetOutputPort())
    writer.SetFileName(output)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()

if __name__ == '__main__':

    # root = r'C:\data\pre-post-paired-40-send-copy-0321'
    # _, sub_list, _ = next(os.walk(root))
    # for sub in sub_list:
    #     main(sub)
    main(*sys.argv[1:])