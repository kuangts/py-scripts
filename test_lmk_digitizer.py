import os, glob
from tools import *
from tools.landmark import *


if __name__ == '__main__':

    ncase = 'n0001'

    nifti_pre = glob.glob(os.path.join(r'C:\data\pre-post-paired-40-send-1122', ncase, '*-pre.nii.gz'))
    assert len(nifti_pre) == 1, 'check patient dir'
    nifti_pre = nifti_pre[0]


    img = imagedata_from_nifti(nifti_pre)
    mask = threshold_imagedata(img, foreground_threshold)
    polyd_pre = polydata_from_mask(mask)
    lmk_file_path = os.path.join(r'C:\data\pre-post-paired-soft-tissue-lmk-23', ncase, 'skin-pre-23-test.csv')

    sel = Digitizer()
    sel.initialize(polyd_pre, lmk_file_path, override=True)
    sel.start()