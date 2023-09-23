from patientdata import PatientData
import numpy as np

data = PatientData()
data.import_image(r'C:\data\pre-post-paired-40-send-1122\n0001\20110425-pre.nii.gz')
# np.save(r'test/asdf.npy', data.image, allow_pickle=False)
# data = PatientData()
# data.load_image(r'test/asdf.npy')
data.import_landmark(r'C:\data\pre-post-paired-40-send-1122\n0001\mandible_landmark.txt', automated=False)
data.mask.append(
    data.create_mask_with_threshold(325, 1250, 'soft tissue')
)
data.mask.append(
    data.create_mask_with_threshold(1250, 4096, 'bone')
)
data.test_show()


