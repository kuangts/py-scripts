import os, enum, time, csv, shutil
from tools import dicom
from tools.polydata import *
from tools.image import *
from tools.dicom import *
from tools.ui import *
from vtk import (
    VTK_DOUBLE,
    vtkDecimatePro,
    vtkCleanPolyData
)

# program parameters
class SHOW_RESULT_LEVEL(enum.IntEnum):
    INTERMEDIATE=0
    IMPORTANT=1
    FINAL=2
    NONE=3

class SAVE_RESULT_LEVEL(enum.IntEnum):
    USEFUL=0
    MINIMUM=1
    NONE=2


def record_entry(sheet_file:str, new_entry:dict):

    with open(sheet_file, 'r') as ff:
        reader = csv.DictReader(ff)
        fieldnames = reader.fieldnames
    
    if not all([f in fieldnames for f in new_entry]):
        raise ValueError('unrecognized field')

    fields = dict.fromkeys(fieldnames, '')
    fields.update(new_entry)
    fields = {k:v for k,v in fields.items() if k in fieldnames}

    with open(sheet_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(fields)

    return None


start_time = time.perf_counter()

# in order to create pre-post-paired case, we need the following data

## in and out
pre_op_dicom_dir = r'P:\eFace Patient Data\for Daeseung\Shanghai_2015Dec-5th\SHANGHAI  9HhospitalData2015-11-27\zhuzewei\2014.12.15 pre\PAT00001\STD00001\SER00003'
post_op_dicom_dir = r'P:\eFace Patient Data\for Daeseung\Shanghai_2015Dec-5th\SHANGHAI  9HhospitalData2015-11-27\zhuzewei\2318391 2015.9.23post\2990161'
save_dir = r'C:\data\pre-post-paired-with-dicom'
registry_file = 'all_modified.csv'
sub = 'n0072'

## calculate the post->pre global registration
### either a previously calculated transformation 
pre_post_transform_file = r''
### or a registered post op part to calculate transformation now
post_op_reg_file = r'P:\eFace Patient Data\for Daeseung\Shanghai_2015Dec-5th\SHANGHAI  9HhospitalData2015-11-27\zhuzewei\STL\ZHU ZE WEI_post_ct_skull_001.stl'
post_op_reg_part = 'bone'

## movement for each segment
pre_op_seg_dir = r''
post_op_seg_dir = r''

# check file
check_file = rf'C:\OneDrive\FEM Mesh Generation\Cases\{sub}\maxilla_surface.stl'

# calculate polydata from image
## read from dicom and transform to vtk
img_pre = dicom.read(pre_op_dicom_dir)
img_post = dicom.read(post_op_dicom_dir)

info_pre = {}
info_pre['Subject ID'] = sub
info_pre['Series ID'] = 'pre'
info_pre['Name'] = img_pre.info['0010|0010'] # Patient's Name
info_pre['Sex'] = img_pre.info['0010|0040'] # Patient's Sex
info_pre['DoB'] = img_pre.info['0010|0030'] # Patient's Birth Date
info_pre['Study Date'] = img_pre.info['0008|0020'] # Study Date
info_pre['Age'] = img_pre.info['0010|1010'] # Patient's Age
info_pre['DICOM'] = pre_op_dicom_dir

info_post = {}
info_post['Subject ID'] = sub
info_post['Series ID'] = 'post'
info_post['Name'] = img_post.info['0010|0010'] # Patient's Name
info_post['Sex'] = img_post.info['0010|0040'] # Patient's Sex
info_post['DoB'] = img_post.info['0010|0030'] # Patient's Birth Date
info_post['Study Date'] = img_post.info['0008|0020'] # Study Date
info_post['Age'] = img_post.info['0010|1010'] # Patient's Age
info_post['DICOM'] = post_op_dicom_dir

img_pre = imagedata_from_sitk(img_pre)
img_post = imagedata_from_sitk(img_post)

# calculate post->pre transformation
if not pre_post_transform_file:
    # mask and polydata
    th = bone_threshold if post_op_reg_part == 'bone' else foreground_threshold
    pre_polyd = polydata_from_mask(threshold_imagedata(img_pre, th))
    post_polyd = polydata_from_mask(threshold_imagedata(img_post, th))
    post_reg_polyd = polydata_from_stl(post_op_reg_file)

    cleaner = vtkCleanPolyData()
    cleaner.SetTolerance(.01)
    cleaner.SetInputData(post_polyd)
    cleaner.Update()
    post_pc = cleaner.GetOutput()
    
    cleaner = vtkCleanPolyData()
    cleaner.SetTolerance(.01)
    cleaner.SetInputData(post_reg_polyd)
    cleaner.Update()
    post_reg_pc = cleaner.GetOutput()

    post_pc_np = vtk_to_numpy(post_pc.GetPoints().GetData())
    post_reg_pc_np = vtk_to_numpy(post_reg_pc.GetPoints().GetData())

    T_init = np.eye(4)
    T_init[:-1,-1] = -post_pc_np.mean(axis=0) + post_reg_pc_np.mean(axis=0)
    post_pc_np = np.hstack((post_pc_np,np.ones((post_pc_np.shape[0],1)))) @ T_init.T
    post_pc_np = post_pc_np[:,:-1]
    T = icp(post_pc_np, post_reg_pc_np) @ T_init
    # T = np.eye(4)
    post_polyd_reg = transform_polydata(post_polyd, T)

    # visualize post->pre registration
    w = Window()
    a = w.add_polydata(post_reg_polyd)
    a.GetProperty().EdgeVisibilityOff()
    a = w.add_polydata(post_polyd_reg)
    a.GetProperty().EdgeVisibilityOff()
    w.initialize()
    w.start()
    
    # visualize pre and post agreement
    w = Window()
    a = w.add_polydata(pre_polyd)
    a.GetProperty().EdgeVisibilityOff()
    a = w.add_polydata(post_polyd_reg)
    a.GetProperty().EdgeVisibilityOff()
    w.initialize()
    w.start()

    # visualize agreement with existing stls
    if check_file:

        w = Window()
        a = w.add_polydata(pre_polyd)
        a.GetProperty().EdgeVisibilityOff()
        a.GetProperty().SetColor(1,0,0)
        a.GetProperty().SetOpacity(.8)
        a = w.add_polydata(polydata_from_stl(check_file))
        a.GetProperty().EdgeVisibilityOff()
        a.GetProperty().SetColor(0,1,0)
        a.GetProperty().SetOpacity(.8)
        w.initialize()
        w.start()
    pre_post_transform = T

else:

    pre_post_transform = np.genfromtxt(pre_post_transform_file)

registry = os.path.join(save_dir, registry_file)
temp_registry = os.path.join(save_dir, '~'+registry_file)

if os.path.exists(temp_registry):
    os.remove(temp_registry)

shutil.copyfile(registry, temp_registry)

# make dir
sub_dir = os.path.join(save_dir, sub)
os.makedirs(sub_dir, exist_ok=True)

# save pre-op image
write_imagedata_to_nifti(img_pre, os.path.join(sub_dir, info_pre['Study Date']+'-pre.nii.gz'))
record_entry(temp_registry, info_pre)

# save post-op image
write_imagedata_to_nifti(img_post, os.path.join(sub_dir, info_post['Study Date']+'-post.nii.gz'))
record_entry(temp_registry, info_post)

# save post->pre registration
np.savetxt(os.path.join(sub_dir, sub+'.tfm'), pre_post_transform, '%.8f')

# save post->pre registration if pre-op origin is reset
t = np.eye(4)
t[:-1,-1] = -np.array(img_pre.GetOrigin())
np.savetxt(os.path.join(sub_dir, sub+'_reset_origin.tfm'), t, '%.8f')

# save pre-op skin, transformed post-op skin, and pre-op bone (for landmarking)
write_polydata_to_stl(
    polydata_from_mask(
        threshold_imagedata(img_pre, foreground_threshold)
        ), 
    os.path.join(sub_dir, 'pre_skin.stl')
    )

write_polydata_to_stl(
    polydata_from_mask(
        threshold_imagedata(img_pre, bone_threshold)
        ), 
    os.path.join(sub_dir, 'pre_bone.stl')
)

write_polydata_to_stl(
    transform_polydata(
        polydata_from_mask(
            threshold_imagedata(img_post, foreground_threshold)
        ),
        pre_post_transform
    ),
    os.path.join(sub_dir, 'post_skin.stl') 
)

if os.path.exists(registry):
    os.remove(registry)

shutil.copyfile(temp_registry, registry)

print(time.perf_counter()-start_time, 'seconds')