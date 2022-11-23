from SimpleITK.SimpleITK import *
from SimpleITK.SimpleITK import _SetImageFromArray
from SimpleITK.extra import *
from SimpleITK._version import __version__

import numpy, dicom, pyvista
import open3d as o3d
from scipy.ndimage import binary_closing
from collections import namedtuple
from collections.abc import Sequence
from copy import deepcopy

Threshold = namedtuple('Threshold',('lo','hi'))
threshold_preset = {
    'bone':Threshold(1250,4095),
    'soft tissue':Threshold(324,1249),
    'all':Threshold(324,4095)
}



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

    @classmethod
    def read(cls, *args):
        return cls(ReadImage(*args))

    @classmethod
    def from_array(cls, arr, copy_info_from=None):
        img = GetImageFromArray(arr.transpose((2,1,0)), isVector=None)
        if copy_info_from:
            img.CopyInformation(copy_info_from)
        return cls(img)

    @classmethod
    def from_gdcm(cls, dicom_series_file_or_dir):
        img = dicom.read(dicom_series_file_or_dir)
        new_img = cls(img)
        if hasattr(img, 'info'):
            setattr(new_img, 'info', img.info)
        return new_img

    def write(*args):
        return WriteImage(*args)

    def write_gdcm(*args):
        return dicom.write(*args)

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

        return self.__class__(resampler.Execute(self))


    def create_model(self, threshold=(), new_spacing=None, 
            smoothing_lambda=lambda m:m.filter_smooth_laplacian(number_of_iterations=5,lambda_filter=.5),
            decimation_lambda=lambda m:m.simplify_quadric_decimation(target_number_of_triangles=100_000),
            ):

        # param threshold looks like this: Threshold(1250,4095) | ( Threshold(1250,4095), ... )
        # each element --> one output model

        if new_spacing is not None:
            def_val = 0
            if hasattr(self, 'info') and '0028|1052' in self.info:
                def_val = self.info['0028|1052']
            else:
                def_val = 0
            img = self.resample(spacing=new_spacing, defaultPixelValue=def_val)
            model = img.create_model(threshold=threshold, new_spacing=None, smoothing_lambda=smoothing_lambda, decimation_lambda=decimation_lambda)
            return model

        if isinstance(threshold, str):
            if threshold in threshold_preset:
                threshold = threshold_preset[threshold]

        if isinstance(threshold, str) or not isinstance(threshold, Sequence):
            raise ValueError('wrong threshold')

        if isinstance(threshold, Threshold):
            # BODY OF THE MOETHOD
            arr = GetArrayViewFromImage(self)
            arr = binary_closing((arr>=threshold.lo) & (arr<=threshold.hi))

            gd = pyvista.UniformGrid(
                dims=self.GetSize(),
                spacing=self.GetSpacing(),
                origin=self.GetOrigin(),
            )
            m = gd.contour([.5], arr.flatten(), method='marching_cubes')
            m_o3d = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(m.points),
                    triangles=o3d.utility.Vector3iVector(m.faces.reshape(-1,4)[:,1:])
                )

            if smoothing_lambda is not None:
                m_o3d = smoothing_lambda(m_o3d)

            if decimation_lambda is not None:
                m_o3d = decimation_lambda(m_o3d)

            m_o3d.compute_triangle_normals()
            m_o3d.compute_vertex_normals()
            # END BODY OF THE MOETHOD

            return m_o3d

        else:
            result = []
            for th in threshold:
                print(th)
                result.append(
                    self.create_model(threshold=th, new_spacing=None, smoothing_lambda=smoothing_lambda, decimation_lambda=decimation_lambda)
                )
            return result


