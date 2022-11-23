from collections.abc import Sequence
from urllib.request import urlopen
from threading import Thread
from xml.etree import ElementTree as ET
import os, re, codecs, csv, shutil, time
import SimpleITK as sitk
import numpy as np

__ALL__ = []


class Tag(list):

    @property
    def tag(self, conj='|'):
        return conj.join(self[0:2])

    @property
    def name(self):
        return self[2]

    def __repr__(self):
        return self.tag + f'({self[3]}): ' + self.name


# def all_tags(self, return_str_or_tuple='str', conj='|'):
# 	if return_str_or_tuple == 'str':
# 		return [ conj.join(t[0:2]) for t in table]
# 	else:
# 		return [ (t[0],t[1]) for t in table]

# def all_names(self):
# 	return [ t[2] for t in table]

class Info(dict):
    """container of dicom info"""

    def __init__(self, *args, check_keys=False, **kwargs):
        if check_keys:
            super().__init__({Dictionary[k].tag: v for k, v in dict(*args, **kwargs).items()})
        else:
            super().__init__(*args, **kwargs)

    def tags(self):
        return list(self.keys())

    def names(self):
        return [Dictionary[k].name for k in self.keys()]

    def __contains__(self, item):
        return super().__contains__(Dictionary[item].tag)

    def __getitem__(self, item):
        return super().__getitem__(Dictionary[item].tag)

    def __setitem__(self, item, val):
        return super().__setitem__(Dictionary[item].tag, val)

    def __repr__(self):
        return '\n'.join([f'{k} -- {Dictionary[k].name}: \n\t{v}' for k, v in self.items()])

    def filter(self, tags_to_keep):
        # return a generic dictionary, with keys from tags_to_keep and present
        self_dict = {}
        for k in tags_to_keep:
            try:
                self_dict[k] = self[Dictionary[k]]
            except Exception as e:
                print(e)
                continue
        return self_dict

    def write(self, to_registry, fieldnames=[]):
        # fieldnames is ignored when there's existing data
        data = []
        if not len(self):
            print('info is empty')
            return
        if not len(fieldnames):
            fieldnames = list(Dictionary.tags_for_record)
        to_registry = os.path.realpath(os.path.normpath(os.path.expanduser(to_registry)))
        temp_copy = ''

        if os.path.isfile(to_registry):
            temp_copy = os.path.join(os.path.dirname(to_registry), '_' + os.path.basename(to_registry))
            try:
                shutil.copy(to_registry, temp_copy)
            except Exception as e:
                print(f'cannot write to {to_registry}')
                print(e)
                return
            # read fieldnames, check duplicate
            with open(to_registry, 'r') as f:
                rows = f.readlines()
                if len(rows):
                    data = list(csv.reader(rows))
                    fieldnames, *data = data
                    data = [dict(zip(fieldnames, d)) for d in data]

        # check duplicate is not implemented yet

        data += [self.filter(fieldnames)]

        try:
            with open(to_registry, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for d in data:
                    writer.writerow(d)
        except:
            print('csv write failed')
            if temp_copy:
                shutil.copy(temp_copy, to_registry)
        finally:
            if temp_copy:
                os.remove(temp_copy)

def load_table(table):
    # part06.html and dicom.css are downloaded from https://dicom.nema.org/medical/dicom/current/output/html/ with no modification
    local_file = rf'{__file__}\..\nema-dicom-part06\part06.html'
    if os.path.isfile(local_file):
        with codecs.open(local_file, mode='r', encoding='utf-8') as f:
            t = f.read()
    else:
        print('loading dicom dictionary from https://dicom.nema.org/medical/dicom/current/output/html/part06.html')
        with urlopen('https://dicom.nema.org/medical/dicom/current/output/html/part06.html') as u:
            t = u.read().decode('utf-8')
    body = re.findall(r'<body>.*</body>', t, re.MULTILINE | re.DOTALL)[0]
    tree = ET.fromstring(body)

    for tr in tree.find(".//*[@id='table_6-1']/..//*[@class='table-contents']/table/tbody"):
        entry = []
        row = tr.findall('./td/p')
        for d in [0, 1, 3]:
            try:
                entry += [row[d].find('span').text]
            except:
                try:
                    entry += [row[d].find('a').tail]
                except:
                    entry += [row[d].text]
        table += [Tag([*entry.pop(0).lower().strip('()').split(','),
                        *map(str, entry)])]  # only use Tag.__init__() here at the creation

class Dictionary:
    tags_for_record = [
        "Study ID",  # example FACEPRED
        "Clinical Trial Subject ID",
        # example (this is the anonymized name of the patient, throughout each study) FACEPRED-001
        "Clinical Trial Series ID",  # example (64 chars max) FACEPRED-001-pre
        "Clinical Trial Series Description",  # example (64 chars max) Facial Prediction Case 001 Pre-operative CT Scan
        "Patient's Name",
        "Patient's Sex",
        "Patient's Age",
        "Patient's Birth Date",
        "Instance Creation Date",
        "Study Date",
        "Series Date",
        "Patient Comments",
    ]

    def __init__(self):
        setattr(self, 'table', [])
        self._t = Thread(target=load_table, args=(self.table,))
        self._t.start()

    def __repr__(self):
        return '\n'.join([t.__repr__() for t in self.lookup_table])

    def __contains__(self, item):
        return self[item] is not None

    def __getitem__(self, item):

        table = self.lookup_table
        if isinstance(item, Tag):
            return item

        translation_table = {
            'Study ID': ('study',),
            'Clinical Trial Subject ID': ('subject', 'subjectid', 'id'),
            'Clinical Trial Series ID': ('series', 'seriesid'),
            "Patient's Name": ('name', 'patient'),
            "Patient's Age": ('age',),
            "Patient's Sex": ('sex', 'gender'),
            "Patient's Birth Date": ('birthdate', 'birthday', 'dob', 'dateofbirth'),
            'Patient Comments': ('description', 'comments', 'comment'),
        }  # use this dictionary case-insensitively and only with alphanumeric characters

        if isinstance(item, str):
            if '|' in item:
                item_split = item.split('|')
            elif ',' in item:
                item_split = item.split(',')
            else:
                item_split = []
            if len(item_split) == 2 and len(item_split[0]) == 4 and len(item_split[1]) == 4:
                item = item_split
            else:
                for k, v in translation_table.items():
                    if ''.join(filter(str.isalnum, item)).lower() in v:
                        item = k
                        break

        if isinstance(item, str):
            for i in range(len(table)):
                if item.lower() == table[i][2].lower():
                    return table[i]
            for i in range(len(table)):
                it = ''.join(filter(str.isalnum, item.lower()))
                tb = ''.join(filter(str.isalnum, table[i][2].lower()))
                if it == tb:
                    return table[i]

        if isinstance(item, Sequence) and len(item) == 2:
            for i in range(len(table)):
                if item[0].lower() == table[i][0] and item[1].lower() == table[i][1]:
                    return table[i]
            else:
                return Tag([*item, 'NA', 'NA'])
        raise ValueError(f'{item} is not in dicom dictionary')

    @property
    def lookup_table(self):
        if hasattr(self, '_t') and self._t.is_alive:
            self._t.join()
            delattr(self, '_t')
            return self.table
        else:
            if not hasattr(self,'table'):
                setattr(self, 'table', [])
                load_table(self.table)
            return self.table

    def search(self, *list_of_necessary_words):
        result = []
        for t in self.lookup_table:
            if all([any([w in i for i in t]) for w in list_of_necessary_words]):
                result += [t]
        return result

    def vr(self, *item): # value representation
        table = self.lookup_table
        if len(item) == 1:
            item = item[0]
            assert isinstance(item, str)
            for i in range(len(table)):
                if item == table[i][2]:
                    return table[i][3]
        elif len(item) == 2:
            assert isinstance(item[0], str) and isinstance(item[1], str)
            for i in range(len(table)):
                if item[0] == table[i][0] and item[1] == table[i][1]:
                    return table[i][3]
        raise ValueError(f'{item} is not in dicom dictionary')

def read_info(file_or_dir, return_filenames=False):
    def _get_info(file):
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(file)
        file_reader.LoadPrivateTagsOn()
        file_reader.ReadImageInformation()
        info = {k: file_reader.GetMetaData(k) for k in file_reader.GetMetaDataKeys()}
        return Info(info)

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

    info, dicom_names = read_info(file_or_dir, return_filenames=True)

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


def anonymize(dcm_dir_in, dcm_dir_out, reset_origin=False, **anon_info):
    # anon_info can include study, subject, series, description, ...
    # program will try to understand those terms and assign to the correct tag
    img = read(dcm_dir_in, return_info=True)
    info = img.info
    img.info = Info(anon_info, check_keys=True)
    if reset_origin:
        img.SetOrigin((0.0, 0.0, 0.0))
    write(img, dcm_dir_out)
    return info


def close_enough(d1, d2, *kwargs):
    img1 = read(d1)
    img2 = read(d2)
    equality = (np.allclose(img1.GetSize(), img2.GetSize())) \
               and (np.allclose(img1.GetOrigin(), img2.GetOrigin())) \
               and (np.allclose(img1.GetSpacing(), img2.GetSpacing())) \
               and (np.allclose(img1.GetDirection(), img2.GetDirection())) \
               and (np.allclose(sitk.GetArrayFromImage(img1), sitk.GetArrayFromImage(img2)))
    return equality


Dictionary = Dictionary()