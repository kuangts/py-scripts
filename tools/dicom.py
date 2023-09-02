import os, time
import SimpleITK as sitk

def _read_info(file_or_dir, return_filenames=False):
    def _get_info(file):
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(file)
        file_reader.LoadPrivateTagsOn()
        file_reader.ReadImageInformation()
        info = {k: file_reader.GetMetaData(k) for k in file_reader.GetMetaDataKeys()}
        return info

    file_or_dir = os.path.normpath(os.path.realpath(file_or_dir))
    if not os.path.exists(file_or_dir):
        raise ValueError(f'{file_or_dir} is not a valid path')
    if os.path.isfile(file_or_dir):
        info = _get_info(file_or_dir)
        if return_filenames:
            return (
                info,
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(os.path.dirname(file_or_dir), info['0020|000e']),
            )
        else:
            return info
    elif os.path.isdir(file_or_dir):
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_or_dir)
        if not series_IDs:
            raise ValueError("directory \"" + file_or_dir + "\" does not contain a DICOM series.")
        file_names = [sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_or_dir, id) for id in series_IDs]
        dicom_names = file_names[0]
        if len(series_IDs) > 1:
            print('WARNING: more than one series found')
            for i,id in enumerate(series_IDs):
                print(f'  {i:>2} - {id} - {len(file_names[i])} slices')
            try:
                dicom_names = file_names[int(input('Select series to read...\n'))]
            except:
                raise ValueError('Read failed')
        info = _get_info(dicom_names[0])
        if return_filenames:
            return (info, dicom_names)
        else:
            return info
    else:
        raise ValueError(f'{file_or_dir} invalid path')


def read(file_or_dir):

    info, dicom_names = _read_info(file_or_dir, return_filenames=True)

    reader = sitk.ImageSeriesReader()
    reader.SetImageIO("GDCMImageIO")
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    
    if '0028|1052' in info and '0028|1053' in info:
        img = (img - info['0028|1052']) * info['0028|1053']
        info.pop('0028|1052')
        info.pop('0028|1053')
    
    setattr(img, 'info', info)
    return img


def write(img, dcm_dir, file_name_format='{:04}.dcm'):
    dcm_dir = os.path.normpath(os.path.realpath(os.path.expanduser(dcm_dir)))
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

    writer = sitk.ImageFileWriter()
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
    direction = img.GetDirection()
    # Tags shared by the series.
    series_tag_values = {
        "0008|0012":modification_date,
        "0008|0013":modification_time,
        "0008|0060": "CT",
        # Setting the type to CT so that the slice location is preserved and the thickness is carried over.
        "0008|0008": "DERIVED\\SECONDARY",  # Image Type
        "0020|000e": "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time, # Series Instance UID
        "0020|0037": '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                         direction[1], direction[4], direction[7]))),  # Image Orientation # (Patient)
    }
    if hasattr(img, 'info'):
        series_tag_values.update(img.info)
    # Write slices to output directory
    for i in range(img.GetDepth()):
        image_slice = img[:, :, i]
        #   Image Position (Patient) --  also determines the spacing between slices
        series_tag_values["0020|0032"] = '\\'.join(map(str, img.TransformIndexToPhysicalPoint((0, 0, i))))
        #   Instance Number
        series_tag_values["0020|0013"] = str(i)
        #   set
        for k, v in series_tag_values.items():
            image_slice.SetMetaData(k, v)
        # Write to the output directory and add the extension dcm, to force
        # writing in DICOM format.
        writer.SetFileName(os.path.join(dcm_dir, file_name_format.format(i + 1)))
        writer.Execute(image_slice)



