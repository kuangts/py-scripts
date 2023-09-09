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
if 'winrar' not in os.environ['PATH'].lower() and os.path.isdir(r'C:\Program Files\WinRAR'):
    os.environ['PATH'] = os.pathsep.join((os.environ['PATH'], r'C:\Program Files\WinRAR'))
import shutil
from tempfile import mkdtemp
import numpy as np
from rarfile import RarFile


def rar_read_bin_arr(rar_file_path, name, dtype):
    cwd, d = os.getcwd(), mkdtemp()
    os.chdir(d)
    try:
        os.system(rf'unrar e -idq {rar_file_path} {name}')
        bytes = np.fromfile(name, dtype=dtype)
        os.chdir(cwd)
    except Exception as e:
        print('could not read binary')
        raise e
    finally:
        shutil.rmtree(d)
    
    return bytes


def rar_save_bin_arr(rar_file_path, name, bytes:np.ndarray):
    cwd, d = os.getcwd(), mkdtemp()
    os.chdir(d)
    try:
        bytes.tofile(name)
        os.system(rf'rar a -idq {rar_file_path} {name}')
        os.chdir(cwd)
    except Exception as e:
        print('could not write binary')
        raise e
    finally:
        shutil.rmtree(d)

    return None
        

def rar_read_text(rar_file_path, name):
    cwd, d = os.getcwd(), mkdtemp()
    os.chdir(d)
    try:
        os.system(rf'unrar e -idq {rar_file_path} {name}')
        with open(name, 'r') as f:
            text = f.read()
        os.chdir(cwd)
    except Exception as e:
        print('could not read text')
        raise e
    finally:
        shutil.rmtree(d)
    
    return text


def rar_save_text(rar_file_path, text:str, name):
    cwd, d = os.getcwd(), mkdtemp()
    os.chdir(d)
    try:
        with open(name, 'w') as f:
            f.write(text)
        os.system(rf'rar a -idq {rar_file_path} {name}')
        os.chdir(cwd)
    except Exception as e:
        print('could not write text')
        raise e
    finally:
        shutil.rmtree(d)

    return None


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
    """this class intermidiates between CASS files and python arrays/strings resulted from server-side computation
    each instance represents the RAR compressed file used to store Computer Aided Surgical Simulation (CASS)
    some methods use Python binding, others system calls due to speed concerns
    this class is also a context manager, so, with CASS ...
    """

    def __init__(self, file, mode="r", charset=None, info_callback=None, crc_check=True, errors="stop"):
        super().__init__(file, mode, charset, info_callback, crc_check, errors)
        self._info = {}
        return None


    @property
    def path(self):
        return self._rarfile
    

    @property
    def dicom_info(self):
        if 'dicom' not in self._info:
            self._info['dicom'] = dict()

            info_str = rar_read_text(self.path, 'Patient_info.bin')
            info = info_str.split(',')
            self._info['dicom']['name'] = info[0]
            self._info['dicom']['study date'] = info[1]
            self._info['dicom']['sex'] = info[2]

            info_str = rar_read_text(self.path, 'Patient_info_new2.bin')
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
            info_str = rar_read_text(self.path, 'Patient_info.bin')
            info = info_str.split(',')
            self._info['image'] = dict()
            self._info['image']['shape'] = tuple(map(int,info[7:4:-1]))
            self._info['image']['spacing'] = tuple(map(float,info[8:11]))
            # not finished

        return self._info['image']


    @property
    def mask_info(self):
        if 'masks' not in self._info:
            info_str = rar_read_text(self.path, 'Mask_Info.bin')
            info = [s.split(',') for s in info_str.strip(';').split(';')]
            self._info['masks'] = []
            for i in range(int(info.pop(0)[0])):
                self._info['masks'].append({
                        'name': info[i][0],
                        'threshold':info[i][1:3],
                        'color':info[i][3:6],
                    })
            # not finished
                
        return self._info['masks']


    @property
    def image(self):
        if not hasattr(self, '_image'):
            bytes = rar_read_bin_arr(self.path, 'Patient_data.bin', np.int16)
            self._image = bytes.reshape(self.image_info['shape'][::-1])[::-1,::-1,:]  
        
        return self._image


    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            masks = []
            for i in range(len(self.mask_info)):
                bytes = rar_read_bin_arr(self.path, f'{i}.bin', np.int16)
                arr = run_length_decode(bytes, self.image_info['shape'][::-1])[::-1,::-1,:]
                masks.append(arr)
            self._masks = np.array(masks, dtype=bool)
        return self._masks


    def pack_mask_info_str(self):
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


    def write(self, items_list):
        if 'image' in items_list:
            rar_save_bin_arr(self.path, 'Patient_data.bin', 
                                np.ravel(self.image[::-1,::-1,:].astype(np.int16), order='C'))
        # not finished : need to update image info too

        if 'masks' in items_list:
            for i in range(len(self.mask_info)):
                rar_save_bin_arr(self.path, f'{i}.bin', 
                                    run_length_encode(self.masks[i,::-1,::-1,:]))
            rar_save_text(self.path, self.pack_mask_info_str(), 'Mask_Info.bin')

        # not finished
