import os, glob
from vtk import vtkNIFTIImageReader, vtkFlyingEdges3D
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy.spatial  import KDTree
from landmark import LandmarkDict
from image import Image


subs0 = r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\jungwook23'
subs1 = r'P:\pre-post-paired'
dir_save = r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\jungwook23-add'
os.makedirs(dir_save, exist_ok=True)
subs = os.listdir(subs0)


for sub in subs:
    print(sub)
    pre_file = glob.glob(os.path.join(subs0, sub, r'*-pre-23.csv'))[0]
    pre_normal_file = pre_file.replace('.csv', '-normal.csv')
    post_file = glob.glob(os.path.join(subs0, sub, r'*-post-post23.csv'))[0]
    pre_file_add = glob.glob(os.path.join(subs1, sub, r'*-pre-pre23.csv'))[0]
    post_file_add = glob.glob(os.path.join(subs1, sub, r'*-post-post23.csv'))[0]
    lmk_pre = LandmarkDict.from_text(pre_file)
    nml_pre = LandmarkDict.from_text(pre_normal_file)
    lmk_post = LandmarkDict.from_text(post_file)
    lmk_pre_add = LandmarkDict.from_text(pre_file_add)
    lmk_post_add = LandmarkDict.from_text(post_file_add)
    img_pre = os.path.join(r'C:\data\pre-post-paired-40-send-1122',sub,os.path.basename(pre_file).replace('-23.csv', '.nii.gz'))
    img_post = os.path.join(r'C:\data\pre-post-paired-40-send-1122',sub,os.path.basename(post_file).replace('-post23.csv', '.nii.gz'))

    reader = vtkNIFTIImageReader()
    reader.SetFileName(img_pre)
    reader.Update()
    out = reader.GetOutput()
    extractor = vtkFlyingEdges3D()
    extractor.SetInputData(out)
    extractor.SetValue(0, 324)
    extractor.Update()
    out = extractor.GetOutput()
    kdtree_pre = KDTree(vtk_to_numpy(out.GetPoints().GetData()))

    reader = vtkNIFTIImageReader()
    reader.SetFileName(img_post)
    reader.Update()
    out = reader.GetOutput()
    extractor = vtkFlyingEdges3D()
    extractor.SetInputData(out)
    extractor.SetValue(0, 324)
    extractor.Update()
    out = extractor.GetOutput()
    kdtree_post = KDTree(vtk_to_numpy(out.GetPoints().GetData()))

    copied = lmk_pre.select(['En-R','En-L','Ex-R','Ex-L'])
    mapped = lmk_pre.select(["Gb'","Zy'-R","Zy'-L"])

    o1 = Image.read(img_pre).GetOrigin()
    o2 = Image.read(img_post).GetOrigin()
    T = np.genfromtxt(os.path.join(subs1, sub, sub+'.tfm'))



    for k in ["Gb'","Zy'-R","Zy'-L"]:
        point, direction = mapped[k], nml_pre[k]
        d, _ = kdtree_pre.query(point, k=1)
        tr = KDTree(np.hstack((kdtree_pre.data, kdtree_pre.data-np.array([point]))))
        d, id = tr.query([*point, *map(lambda x:x*d, direction)], k=1)
        lmk_pre[k] = tr.data[id,:3].tolist()

    mapped = (mapped + o1) @ np.linalg.inv(T.T) - o2

    for k in ["Gb'","Zy'-R","Zy'-L"]:
        point, direction = mapped[k], nml_pre[k]
        d, _ = kdtree_post.query(point, k=1)
        tr = KDTree(np.hstack((kdtree_post.data, kdtree_post.data-np.array([point]))))
        d, id = tr.query([*point, *map(lambda x:x*d, direction)], k=1)
        lmk_post[k] = tr.data[id,:3].tolist()

    copied = (copied + o1) @ np.linalg.inv(T.T) - o2

    for k in ['En-R','En-L','Ex-R','Ex-L']:
        point = copied[k]
        d, id = kdtree_post.query(point, k=1)
        lmk_post[k] = kdtree_post.data[id].tolist()

    lmk_pre.update(lmk_pre_add)
    lmk_post.update(lmk_post_add)

    lmk_pre = lmk_pre + o1
    lmk_post = lmk_post + o2

    os.makedirs(os.path.join(dir_save, sub), exist_ok=True)

    lmk_pre.write(os.path.join(dir_save, sub, r'skin-pre-23.csv'))
    lmk_post.write(os.path.join(dir_save, sub, r'skin-post-23.csv'))





