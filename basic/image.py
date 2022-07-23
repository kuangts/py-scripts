import numpy
import time, os
from typing import List, Union
from basic import dicom

from SimpleITK.SimpleITK import *
from SimpleITK.SimpleITK import _SetImageFromArray
from SimpleITK.extra import *
from SimpleITK._version import __version__


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

    def write_gdcm(self, dcm_dir, info, file_name_format='{:04}.dcm'):
        dcm_dir = os.path.expanduser(dcm_dir)
        dcm_dir = os.path.normpath(os.path.realpath(dcm_dir))
        if os.path.exists(dcm_dir) and not os.path.isdir(dcm_dir):
            print(f'cannot write to {dcm_dir}, not a directory')
        else:
            os.makedirs(dcm_dir, exist_ok=True)
        if os.listdir(dcm_dir):
            print(f'cannot write to {dcm_dir}, directory not empty')
            return

        # downloaded from simpleitk example 'Dicom Series From Array'
        # at https://simpleitk.readthedocs.io/en/master/link_DicomSeriesFromArray_docs.html
        # modified to always write int16, original huntsfield value

        # Write the 3D image as a series
        # IMPORTANT: There are many DICOM tags that need to be updated when you modify
        #            an original image. This is a delicate operation and requires
        #            knowledge of the DICOM standard. This example only modifies some.
        #            For a more complete list of tags that need to be modified see:
        #                  http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
        #            If it is critical for your work to generate valid DICOM files,
        #            It is recommended to use David Clunie's Dicom3tools to validate
        #            the files:
        #                  http://www.dclunie.com/dicom3tools.html

        writer = ImageFileWriter()
        writer.SetImageIO("GDCMImageIO")
        # Use the study/series/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        # Copy some of the tags and add the relevant tags indicating the change.
        # For the series instance UID (0020|000e), each of the components is a number,
        # cannot start with zero, and separated by a '.' We create a unique series ID
        # using the date and time. Tags of interest:
        direction = self.GetDirection()
        # Tags shared by the series.
        series_tag_values = {

            "0008|0060": "CT", # Setting the type to CT so that the slice location is preserved and the thickness is carried over.
            "0008|0008": "DERIVED\\SECONDARY",  # Image Type
            "0020|000e": "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time,  # Series Instance UID
            "0020|0037": '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                            direction[1], direction[4], direction[7]))), # Image Orientation # (Patient)
        }
        series_tag_values.update(info)
        # Write slices to output directory
        for i in range(self.GetDepth()):
            image_slice = self[:, :, i]
            #   Instance Creation Date
            series_tag_values["0008|0012"] = time.strftime("%Y%m%d")
            #   Instance Creation Time
            series_tag_values["0008|0013"] = time.strftime("%H%M%S")
            # (0020, 0032) image position patient determines the 3D spacing between
            # slices.
            #   Image Position (Patient)
            series_tag_values["0020|0032"] = '\\'.join(map(str, self.TransformIndexToPhysicalPoint((0, 0, i))))
            #   Instance Number
            series_tag_values["0020|0013"] = str(i)
            #   set
            for k,v in series_tag_values.items():
                image_slice.SetMetaData(k,v)
            # Write to the output directory and add the extension dcm, to force
            # writing in DICOM format.
            writer.SetFileName(os.path.join(dcm_dir, file_name_format.format(i+1)))
            writer.Execute(image_slice)

    @classmethod
    def from_array(cls, arr, copy_info_from=None):
        arr = arr.transpose((2,1,0))
        img = GetImageFromArray(arr, isVector=None)
        if copy_info_from:
            img.CopyInformation(copy_info_from)
        return cls(img)

    @classmethod
    def from_gdcm(cls, dicom_series_file_or_dir, return_image=True, return_info=False, **dicomargs):
        if not os.path.exists(dicom_series_file_or_dir):
            return None
        elif os.path.isfile(dicom_series_file_or_dir):
            try:
                file_reader = ImageFileReader()
                file_reader.SetFileName(dicom_series_file_or_dir)
                file_reader.ReadImageInformation()
                dicom_series_file_or_dir = os.path.dirname(dicom_series_file_or_dir) # change this later to finding files within the same series
            except:
                print(dicom_series_file_or_dir, "has no image information")
                return None
        
        series_IDs = ImageSeriesReader.GetGDCMSeriesIDs(dicom_series_file_or_dir)
        if not series_IDs:
            print("ERROR: directory \"" + dicom_series_file_or_dir + "\" does not contain a DICOM series.")
            return None
        print(f'series id {series_IDs}')

        result = ()

        if return_image:
            dicom_names = ImageSeriesReader.GetGDCMSeriesFileNames(dicom_series_file_or_dir, series_IDs[0])
            reader = ImageSeriesReader()
            reader.SetImageIO("GDCMImageIO")
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            img = reader.Execute()
            result += (cls(img),)

        if return_info:
            file_reader = ImageFileReader()
            file_reader.SetFileName(dicom_names[0])
            file_reader.LoadPrivateTagsOn()
            file_reader.ReadImageInformation()
            info = { k:file_reader.GetMetaData(k) for k in file_reader.GetMetaDataKeys() }
            result += (dicom.Info(info),)
        
        if len(result)==1:
            result = result[0]
        return result
    
    @classmethod
    def read(cls, img_path, **initargs):
        return cls(ReadImage(img_path))

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
