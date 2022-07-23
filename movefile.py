import os
import shutil
import basic

original_dicom_dir = r'C:\Users\tmhtxk25\Desktop\FL-dicom'
anonymized_dicom_dir = r'C:\Users\tmhtxk25\Desktop\FL-dicom-anon'
path_to_registry = r'C:\Users\tmhtxk25\Desktop\reg.csv'

os.chdir(original_dicom_dir)
for i in os.listdir('.'):
    for j in os.listdir(rf'..\FL-dicom-anon\{i}'): os.remove(rf'..\FL-dicom-anon\{i}\{j}')
    if not len(os.listdir(i)):
        continue
    img, info = basic.Image.from_gdcm(i, return_info=True)
    print(i, info[basic.DD["Patient's Name"].join()], img['origin'])
    info = basic.dicom_anonymize(i, rf'..\FL-dicom-anon\{i}', reset_origin=True, study_id='FL', subject_id=i, series_id=i+'-1', description=f'Federated Learning {i}')
    basic.write_info(path_to_registry, info)

os.chdir(anonymized_dicom_dir)
for i in os.listdir('.'):
    if not len(os.listdir(i)):
        continue
    img, info = basic.Image.from_gdcm(i, return_info=True)
    print(i, info[basic.DD['Clinical Trial Series ID'].join()], img['origin'])
