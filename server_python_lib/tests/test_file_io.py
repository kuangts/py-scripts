import os, shutil
import SimpleITK as sitk
import numpy as np

from server_python_lib import *


dicom_path=r'C:\data\pre-post-paired-with-dicom\n0001\20110425-pre',
cass_path=r'C:\data\pre-post-paired-with-dicom\n0001\pre.CASS',


# READ IMAGE FROM ORIGINAL DICOM
from tools import dicom
img = dicom.read(dicom_path)
# `img` is a SimpleITK Image object
# to be consistent with AnatomicAligner,
# the intercept and slope from dicom header are applied, if they aren't already
if hasattr(img, 'info'):
    info = img.info    
    if '0028|1052' in info and '0028|1053' in info:
        img = (img - info['0028|1052']) * info['0028|1053']
        info.pop('0028|1052')
        info.pop('0028|1053')

# CREATE BONE SEGMENTATION WITH THRESHOLD
img_arr = sitk.GetArrayFromImage(img)

mask_arr_bone = np.logical_and(img_arr>=1250, img_arr<=4095)
mask_info_bone = dict(
        name='bone',
        threshold=(1250,4095),
        color=(255,128,255),
    )

mask_arr_skin = np.logical_and(img_arr>=324, img_arr<=1249)
mask_info_skin = dict(
        name='skin',
        threshold=(324,1249),
        color=(128,255,128),
    )

# OPEN CASS CREATED WITH THE SAME SERIES
with CASS(cass_path) as f:
    img_arr_cass = f.image
    mask_arr_bone_cass = f.masks[0]
    mask_info_bone_cass = f.mask_info[0]


if np.array_equal(img_arr_cass, img_arr) and np.array_equal(mask_arr_bone, mask_arr_bone_cass):
    print('image and masks are the same')
else:
    print('image and mask are DIFFERENT')

mask_arr_bone[-1,-1,-1] = not mask_arr_bone[-1,-1,-1]
print('after assignment, ')

if np.array_equal(img_arr_cass, img_arr) and np.array_equal(mask_arr_bone, mask_arr_bone_cass):
    print('image and masks are the same')
else:
    print('image and mask are DIFFERENT')

#----------------------------------------------------------#
# at this point, read functions are proven correct
#----------------------------------------------------------#

# COPY CASS, SHOW, AND WRITE TO THE NEW COPY
new_cass_path = os.path.join(os.path.dirname(cass_path), 'new_'+os.path.basename(cass_path))
shutil.copy(cass_path, new_cass_path)
with CASS(new_cass_path) as f:
    f.show(50)
    # add a new mask (soft tissue) too, which does not exist in original copy
    f.add_mask(mask_arr_skin, **mask_info_skin)
    f.write(['image', 'masks'])


# TEST FILE EQUALITY UPON WRITING
arr0 = rar_read_bin_arr(cass_path, 'Patient_data.bin', dtype=np.int16)
arr1 = rar_read_bin_arr(new_cass_path, 'Patient_data.bin', dtype=np.int16)
print(f'images are {"THE SAME"  if np.array_equal(arr0,arr1) else "DIFFERENT"}')
msk0 = rar_read_bin_arr(cass_path, f'0.bin', dtype=np.int16)
msk1 = rar_read_bin_arr(new_cass_path, f'0.bin', dtype=np.int16)
print(f'masks are {"THE SAME"  if np.array_equal(msk0,msk1) else "DIFFERENT"}')


# CHANGE ONE SLICE THEN TEST FILE EQUALITY AGAIN
with CASS(new_cass_path) as f:
    f.image[50,:,:] = -2000
    f.write(['image','masks'])
arr0 = rar_read_bin_arr(cass_path, 'Patient_data.bin', dtype=np.int16)
arr1 = rar_read_bin_arr(new_cass_path, 'Patient_data.bin', dtype=np.int16)
print(f'images are {"THE SAME"  if np.array_equal(arr0,arr1) else "DIFFERENT"}')
msk0 = rar_read_bin_arr(cass_path, f'0.bin', dtype=np.int16)
msk1 = rar_read_bin_arr(new_cass_path, f'0.bin', dtype=np.int16)
print(f'masks are {"THE SAME"  if np.array_equal(msk0,msk1) else "DIFFERENT"}')

#----------------------------------------------------------#
# at this point, write functions are proven correct
#----------------------------------------------------------#

# OPEN NEW CASS COPY IN ANATOMICALIGNER AND SEE THE FAULTY SLICE + NEW SOFT TISSUE MASK



