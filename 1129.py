# read normal ct dicom
import os, glob, csv, shutil
import numpy as np
import dicom
import SimpleITK as sitk

allseries = glob.glob(r'C:\data\SH\Data_from_Shanghai_NormalCT_DICOM\CD*\PA*\ST*\SE*')
slen = []
name = []
info_all = []

for i in allseries:
    info, fnames = dicom.read_info(i, return_filenames=True)
    info_all.append(info)
    slen.append(len(fnames))
    name.append(info['name'].strip())

ind = np.argsort(slen)
seri = {}
info = {}

for i in ind[::-1]:
    if name[i] not in seri:
        seri[name[i]] = allseries[i]
        info[name[i]] = info_all[i]

fieldnames = ["Patient's Name", "Patient ID", "Study ID", "Patient's Age", "Patient's Sex", "Study Date", "DICOM Origin X", "DICOM Origin Y", "DICOM Origin Z", "DICOM Path", "NIFTI Path"]

with open(r'c:\data\SH\registry.csv', 'w', newline='') as reg:
    writer = csv.DictWriter(reg, fieldnames)
    writer.writeheader()
    for name in info:
        img = dicom.read(seri[name])
        row = {f:img.info[f].strip() for f in fieldnames[:6] if f in img.info}
        nii = rf'C:\data\SH\nifti\{img.info["Study ID"].strip()}.nii.gz'
        ori = img.GetOrigin()
        row.update({
            "DICOM Origin X":ori[0],
            "DICOM Origin Y":ori[1],
            "DICOM Origin Z":ori[2],
            "DICOM Path":seri[name],
            "NIFTI Path":nii
        })
        writer.writerow(row)
        img.SetOrigin((0,0,0))
        sitk.WriteImage(img, nii)
        shutil.move(seri[name], rf'c:\data\sh\{img.info["Study ID"].strip()}')


