from SimpleITK.SimpleITK import *
from SimpleITK.SimpleITK import _SetImageFromArray
from SimpleITK.extra import *
from SimpleITK._version import __version__

import numpy, dicom, pyvista
from mesh import TriangleSurface

###### NEEDS MORE TEST ######
class Image(Image):

    ###################################
    # creation and back-storage related
    ###################################

    @property
    def arrayview(self):
        array_view = GetArrayViewFromImage(self)
        return array_view.transpose((2,1,0))

    @property
    def array(self):
        array_view = GetArrayViewFromImage(self)
        return numpy.array(array_view, copy=True).transpose((2,1,0))
        

    def write(self, img_path, imageIO=''):
        return WriteImage(self, img_path, imageIO=imageIO)

    def write_gdcm(self, dcm_dir, info={}):
        dicom.write(self, dcm_dir, info, file_name_format='{:04}.dcm')

    @classmethod
    def from_array(cls, arr, copy_info_from=None):
        arr = arr.transpose((2,1,0))
        img = GetImageFromArray(arr, isVector=None)
        if copy_info_from:
            img.CopyInformation(copy_info_from)
        return cls(img)

    @classmethod
    def from_gdcm(cls, dicom_series_file_or_dir, **dicomargs):
        img, info = dicom.read(dicom_series_file_or_dir, return_image=True, return_info=True, **dicomargs)
        img = cls(img)
        setattr(img, 'info', dicom.Info(info))
        return img

    @classmethod
    def read(cls, img_path, **initargs):
        img = ReadImage(img_path)
        return cls(img)

    def update_array(self, arr):

        ################# DANGER #################
        ################# DANGER #################
        ################# DANGER #################
        # main purpose is to retain the reference to image
        # only use it for this purpose
        # otherwise GetImageFromArray() is a better choice
        # arr must own its data
        # arr.shape must equal self['shape']
        # arr.dtype must be compatible with img.GetPixelIDTypeAsString()
        # a more reliable way to use this method:
        
            # arr = np.empty_like(img.arrayview)
            # arr[arr<0] = 0 # or other ops
            # img.update_array(arr)
        
        _SetImageFromArray(arr.transpose((2,1,0)).tobytes(), self)

    ###################################
    # utility functions
    ###################################

    def __getitem__( self, idx ):
        if isinstance(idx, str) and idx == 'shape':
            return self.GetSize()
        else:
            return super().__getitem__(idx)

    def resample(self,
           shape=(),
           spacing=(),
           origin=(),
           direction=(),
           transform=0,
           interpolator=0,
           outputPixelType=0,
           defaultPixelValue=0.0,
           useNearestNeighborExtrapolator=False):

        if not shape and not spacing:
            print('resampling aborted')
            return self
        if not shape:
            shape = [ int(self['shape'][i]*self['spacing'][i]/spacing[i]) for i in range(len(self['shape'])) ]
        if not spacing:
            spacing = [ self['shape'][i]*self['spacing'][i]/shape[i] for i in range(len(self['shape'])) ]
        resampler = ResampleImageFilter()
        resampler.SetSize(shape)
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputOrigin(origin if origin else self['origin'])
        resampler.SetOutputDirection(direction if direction else self['direction'])
        resampler.SetTransform(transform if transform else Transform())
        resampler.SetInterpolator(interpolator if interpolator else sitkLinear)
        resampler.SetOutputPixelType(outputPixelType if outputPixelType else sitkUnknown)
        resampler.SetDefaultPixelValue(defaultPixelValue)
        resampler.SetUseNearestNeighborExtrapolator(useNearestNeighborExtrapolator)
        print('resampled from {} {} to {} {}'.format(self['spacing'], self['shape'], spacing, shape))
        return __class__(resampler.Execute(self))


def seg2surf(bin_img):
    gd = pyvista.UniformGrid(
        dims=bin_img.GetSize(),
        spacing=bin_img.GetSpacing(),
        origin=bin_img.GetOrigin(),
    )
    m = gd.contour([.5], bin_img.array.transpose([2,1,0]).flatten(), method='marching_cubes')
    return TriangleSurface(m.points, m.faces.reshape(-1, 4)[:,1:])


def seg2surf_vtk(bin_img):

    from dicom2stl.utils.sitk2vtk import sitk2vtk
    import itk, vtk
    bin_img = sitk2vtk(bin_img)
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(bin_img)
    dmc.ComputeNormalsOn()
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()
    meshout = dmc.GetOutput()
    filt = vtk.vtkSmoothPolyDataFilter()
    filt.SetInputData(meshout)
    filt.FeatureEdgeSmoothingOn()
    filt.Update()
    meshout = filt.GetOutput()

    # filt = itk.CuberilleImageToMeshFilter()
    # filt.SetInput(bin_img)
    # filt.SetIsoSurfaceValue(1)
    # filt.Update()
    # meshout = filt.GetOutput()
    # print(meshout)
    # print(type(meshout))

    V = vtk.util.numpy_support.vtk_to_numpy(meshout.GetPoints().GetData())
    F = vtk.util.numpy_support.vtk_to_numpy(meshout.GetPolys().GetData()).astype(int).reshape(-1,4)[:,1:4]

    return TriangleSurface(V,F)

