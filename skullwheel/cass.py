#!/usr/bin/env python
'''
Tianshu Kuang
Houston Methodist Hospital
09/2023

this module contains utility functions for compression, encoding, decoding, and file related operations
in this module, arrays are transposed and flipped to be consistent with ones returned by `SimpleITK.GetArrayFromImage()` as much as possible
whose dimensions are in the reverse of `GetSize()` method on SimpleITK Image object
as a comparison, arrays decoded from binary stream follow the same dimensional order but have the first two dimensions flipped
without indications otherwise, arrays (`arr`, `bytes`, ...) are numpy.ndarray objects
'''

__all__ = ['CASS']

import os
import shutil
from contextlib import contextmanager
from numbers import Number
from tempfile import mkdtemp
from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
from rarfile import RarFile

import dicom

if (not shutil.which('rar') or not shutil.which('unrar')) and os.path.isdir(r'C:\Program Files\WinRAR'):
    os.environ['PATH'] = os.pathsep.join((os.environ['PATH'], r'C:\Program Files\WinRAR'))


# def tempdir(_func=None, *, location=None):
#     import functools
#     '''decorate functions where files need to be copied
#     and those files should exist only for the duration of the function'''
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             decorator.dir = location
#             if decorator.dir is None:
#                 decorator.dir = mkdtemp()
#             elif not os.path.exists(decorator.dir):
#                 os.makedirs(decorator.dir)
#             else:
#                 raise ValueError('path exists')
#             cwd = os.getcwd()
#             os.chdir(decorator.dir)
#             func(*args, **kwargs)
#             os.chdir(cwd)
#             shutil.rmtree(decorator.dir)
#             decorator.dir = None
#             return None
#         return wrapper
#     if _func is None:
#         return decorator
#     else:
#         return decorator(_func)



def run_length_decode(bytes:np.ndarray, shape):
    arr = np.zeros(shape, dtype=np.int16)
    for i in range(0,bytes.size,4):
        arr[
            bytes[i+3],
            bytes[i],
            bytes[i+1]:bytes[i+1]+bytes[i+2]
            ] = 1
        
    return arr


def run_length_encode(arr:np.ndarray):
    mask_index = np.unique(arr[arr!=0])
    assert mask_index.size==1, 'this is not a mask'
    mask_index = mask_index[0]
    seg_list = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            segs = np.diff(np.hstack((0, arr[i,j], 0)))
            for start, end in zip(np.nonzero(segs==1)[0], np.nonzero(segs==-1)[0]):
                seg_list += [ j,start,end-start,i ]
    bytes = np.array(seg_list, dtype=np.int16)

    return bytes






class CASS(RarFile):
    """this class intermidiates between CASS files and python arrays/strings
    each instance represents the RAR compressed file used to store Computer Aided Surgical Simulation (CASS)
    some methods use Python binding, others system calls due to speed concerns
    this class is also a context manager, so, with CASS ...
    """

    IMAGE_FILE_NAME = 'Patient_Data.bin'
    PATIENT_INFO_FILE1_NAME = 'Patient_info.bin'
    PATIENT_INFO_FILE2_NAME = 'Patient_info_new2.bin'
    MASK_FILE_NAME = 'Mask_data.bin'
    MASK_INFO_FILE_NAME = 'Mask_Info.bin'
    THREE_DIM_OBJECT_INFO_FILE_NAME = 'Three_Data_Info.bin'

    List_of_Files = (
        IMAGE_FILE_NAME,
        PATIENT_INFO_FILE1_NAME,
        PATIENT_INFO_FILE2_NAME,
        MASK_FILE_NAME,
        MASK_INFO_FILE_NAME,
        THREE_DIM_OBJECT_INFO_FILE_NAME,
    )


    def __init__(self, file, mode="r", charset=None, info_callback=None, crc_check=True, errors="stop"):
        super().__init__(file, mode, charset, info_callback, crc_check, errors)
        self._info = {}
        return None


    @property
    def path(self):
        return self._rarfile


    @contextmanager
    def temp_files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        try:
            cwd, context_dir = os.getcwd(), mkdtemp()
            os.chdir(context_dir)
            os.system(rf'rar e -idq {self.path} ' + ' '.join(list_of_files))
            print(os.getcwd())
            yield

        finally:
            os.chdir(cwd)
            print(os.getcwd())
            try:
                shutil.rmtree(context_dir)
                # for f in self.list_of_files:
                #     os.remove(os.path.join(self.context_dir, f))
            except Exception as e:
                print(e)



    @contextmanager
    def open_save_dir(self):
        try:
            cwd, context_dir = os.getcwd(), mkdtemp()
            os.chdir(context_dir)
            print(os.getcwd())
            yield

        finally:
            os.system(rf'rar a -idq {self.path} ' + ' '.join(os.listdir(os.getcwd())))
            os.chdir(cwd)
            print(os.getcwd())
            shutil.rmtree(context_dir)



    def cass_open(self):
        with self.temp_files(self.List_of_Files):
            self.load_image_info()
            self.load_image()
            self.load_mask_info()
            self.load_masks()


    def cass_save(self):
        with self.temp_files(self.List_of_Files):
            self.save_image()
            self.save_masks()



    def load_image(self):
        with self.temp_files(self.IMAGE_FILE_NAME):
            data = np.fromfile(self.IMAGE_FILE_NAME, dtype=np.int16)
            shape = self.image_info['shape'][::-1]
            self._image = data.reshape(shape)[::-1,::-1,:]
            return None


    def save_image(self):
        with self.open_save_dir():
            np.ravel(self.image[::-1,::-1,:].astype(np.int16), order='C').tofile(self.IMAGE_FILE_NAME)
            with open(self.PATIENT_INFO_FILE1_NAME,'w') as f:
                f.write(self.image_info_str())



    def load_image_info(self):
        with self.temp_files(self.PATIENT_INFO_FILE1_NAME), open(self.PATIENT_INFO_FILE1_NAME, 'r') as f:
            info_str = f.read()
            info = info_str.split(',')
            self._info['image'] = dict()

            self._info['image']['name'] = info[0]
            self._info['image']['study_date'] = info[1]
            self._info['image']['sex'] = info[2]
            self._info['image']['window_center'] = int(info[3])
            self._info['image']['window_level'] = int(info[4])
            self._info['image']['shape'] = tuple(map(int,info[7:4:-1]))
            self._info['image']['spacing'] = tuple(map(float,info[8:11]))
            self._info['image']['offset'] = int(info[11])
            self._info['image']['center'] = tuple(map(float,info[12:15]))
            self._info['image']['x_window_center'] = int(info[15])
            self._info['image']['x_window_level'] = int(info[16])
            self._info['image']['coronal_window_center'] = int(info[17])
            self._info['image']['coronal_window_level'] = int(info[18])
            self._info['image']['vector_window_center'] = int(info[19])
            self._info['image']['vector_window_level'] = int(info[20])
            self._info['image']['uid'] = info[21]

            return

            # not finished


    def image_info_str(self):
        for i,v in enumerate(self._info['image'].values()):
            if i == 0:
                info_str = v
            elif isinstance(v,str) or isinstance(v,Number):
                info_str += ','+str(v)
            elif isinstance(v, Sequence):
                info_str += ','+ ','.join(map(str,v))

        return info_str



    def load_masks(self):
        num_mask = len(self.mask_info)
        if not num_mask:
            return
        filenames = [f'{i}.bin' for i in range(num_mask)]
        with self.temp_files(filenames):
            masks = []
            for f in filenames:
                data = np.fromfile(f, dtype=np.int16)
                shape = self.image_info['shape'][::-1]
                masks.append(run_length_decode(data, shape)[::-1,::-1,:] )
            self._masks = np.array(masks, dtype=bool)
            return None


    def save_masks(self):
        if not len(self.mask_info):
            return
        with self.open_save_dir():
            for i in range(len(self.mask_info)):
                run_length_encode(self.masks[i,::-1,::-1,:]).tofile(f'{i}.bin')
            if 'masks' in self._info and self._info['masks']:
                with open(self.MASK_INFO_FILE_NAME, 'w') as f:
                    f.write(self.mask_info_str())



    def load_mask_info(self):
        with self.temp_files(self.MASK_INFO_FILE_NAME):
            if not os.path.isfile(self.MASK_INFO_FILE_NAME):
                self._info['masks'] = []
                return

            with open(self.MASK_INFO_FILE_NAME, 'r') as f:
                info_str = f.read()
        info = [s.split(',') for s in info_str.strip(';').split(';')]
        self._info['masks'] = []
        for i in range(int(info.pop(0)[0])):
            self._info['masks'].append({
                    'name': info[i][0],
                    'threshold':info[i][1:3],
                    'color':info[i][3:6],
                })
        # not finished


    # def read_text(self, file):
    #     with self.temp_files([file]), open(file) as f:
    #         return f.read()
            

    @property
    def dicom_info(self):
        if 'dicom' not in self._info:
            self._info['dicom'] = dict()
            with self.temp_files([self.PATIENT_INFO_FILE1_NAME]), open(self.PATIENT_INFO_FILE1_NAME) as f:
                info_str = f.read()

            info = info_str.split(',')
            self._info['dicom']['name'] = info[0]
            self._info['dicom']['name'] = self._info['dicom']['name'].replace(' ','^')
            self._info['dicom']['name'] = self._info['dicom']['name'].replace(',','^')
            self._info['dicom']['study date'] = info[1]
            self._info['dicom']['sex'] = info[2]

            with self.temp_files([self.PATIENT_INFO_FILE2_NAME]), open(self.PATIENT_INFO_FILE2_NAME) as f:
                info_str = f.read()

            info = info_str.split(',')
            self._info['dicom']['id'] = info[0]
            self._info['dicom']['dob'] = info[1]
            self._info['dicom']['age'] = info[2]
            self._info['dicom']['study id'] = info[3]
            # not finished

        return self._info['dicom']


    @property
    def image_info(self):
        if 'image' not in self._info:
            self.load_image_info()

        return self._info['image']


    @property
    def mask_info(self):
        if 'masks' not in self._info:
            self.load_mask_info()
                
        return self._info['masks']


    @property
    def image(self):
        if not hasattr(self, '_image'):
            self.load_image()
        
        return self._image


    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            self.load_masks()
        return self._masks


    def mask_info_str(self):
        segs = []
        segs.append([len(self.mask_info)])
        for v in self.mask_info:
            seg = []
            seg.append(v['name'])
            for m in v['threshold']:
                seg.append(m)
            for m in v['color']:
                seg.append(m)
            seg.append(8.0)
            seg.append(1)
            seg.append(0)
            seg.append(1)
            # figure out what those are
            if 'auto' in v and v['auto']:
                seg.append('Yes')
            segs.append(seg)

        info_string = ';'.join(
            ','.join(map(str,seg)) for seg in segs
        ) + ';'
        
        return info_string


    def show(self, slice):

        import matplotlib.pyplot as plt
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot()
        image_slice = self.image[slice,:,:]
        ax.imshow(image_slice, cmap=plt.cm.gray)
        masks_color = np.zeros(image_slice.shape)
        for i in range(self.masks.shape[0]):
            masks_color[self.masks[i,slice,:,:]] = i+1
        masks_color[masks_color==0] = np.nan
        ax.imshow(masks_color, alpha=.5)
        plt.show()

        return None


    def update_image(self, arr:np.ndarray, **kwargs):
        # update self.image and self.image_info
        if not np.array_equal(self.image.shape, arr.shape):
            print('changing image shape')
        self._image = arr.copy()
        self.image_info['shape'] = self.image.shape[::-1]
        self.image_info.update(kwargs)
        return None


    def update_mask(self, index, arr:np.ndarray, **info):
        # update self.masks and self.mask_info
        if not np.array_equal(self.masks.shape[1:], arr.shape):
            raise ValueError('cannot update mask with a different shape')
        self.masks[index, ...] = arr
        self.mask_info[index].update(info)
        return None


    def add_mask(self, arr:np.ndarray, **info):
        # update self.masks and self.mask_info (name, threhsold and color)
        if not hasattr(self, '_masks') or not self._masks.size:
            self._masks = np.empty((0,*self._image.shape))
        if not np.array_equal(self.masks.shape[1:], arr.shape):
            raise ValueError('cannot add mask with a different shape')
        self._masks = np.append(self.masks, [arr], axis=0)
        self.mask_info.append(info)
        return None


    def delete_mask(self, index):
        if index >= self.masks.shape[0]:
            raise ValueError(f'{index}th mask does not exist')
        self._masks = np.delete(self.masks, index, axis=0)
        self.mask_info.pop(index)
        return None


    @classmethod
    def cass_new(cls, dicom_dir, file_path):
        shutil.copy(r'P:\server_python_lib\tests\blank.CASS', file_path)
        cass_file = cls(file_path)
        cass_file.cass_open()
        img = dicom.read(dicom_dir)
        cass_file._image = sitk.GetArrayFromImage(img)
        info = img.info
        # contains error but tries to appease AA
        cass_file._info['image']['name'] = info['0010|0010']
        cass_file._info['image']['name'] = cass_file._info['image']['name'].replace(' ','^')
        cass_file._info['image']['name'] = cass_file._info['image']['name'].replace(',','^')
        cass_file._info['image']['study_date'] = info['0008|0020']
        cass_file._info['image']['sex'] = info['0010|0040']
        cass_file._info['image']['window_center'] = int(info['0028|1051'])
        cass_file._info['image']['window_level'] = int(info['0028|1050'])
        cass_file._info['image']['shape'] = img.GetSize()[::-1]
        cass_file._info['image']['spacing'] = img.GetSpacing()
        cass_file._info['image']['offset'] = int(info['0028|1052'] if '0028|1052' in info else 0)
        center = np.array((0.,0.,0.))
        center += (np.array(img.GetSize())-1) * img.GetSpacing() * .5
        cass_file._info['image']['center'] = tuple(center.tolist())
        cass_file._info['image']['x_window_center'] = cass_file._info['image']['window_center']
        cass_file._info['image']['x_window_level'] = cass_file._info['image']['window_level']
        cass_file._info['image']['coronal_window_center'] = cass_file._info['image']['window_center']
        cass_file._info['image']['coronal_window_level'] = cass_file._info['image']['window_level']
        cass_file._info['image']['vector_window_center'] = cass_file._info['image']['window_center']
        cass_file._info['image']['vector_window_level'] = cass_file._info['image']['window_level']
        cass_file._info['image']['uid'] = info['0020|000D']  if '0020|000D' in info else '0.0.0'
        cass_file.cass_save()
        return cass_file



if __name__=='__main__':
    # x = CASS(r'P:\Kuang\GO0096RS.CASS')
    # x.cass_open()
    x = CASS.cass_new(r'P:\server_python_lib\tests\go0096',r'P:\server_python_lib\tests\test.CASS')
    mask_arr_skin = np.logical_and(x.image>=324, x.image<=1249)
    mask_info_skin = dict(
            name='skin',
            threshold=(324,1249),
            color=(128,255,128),
        )
    x.add_mask(mask_arr_skin, **mask_info_skin)
    x.cass_save()
    x.cass_open()
    x.show(50)
