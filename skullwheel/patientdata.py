import os, csv, re
from collections.abc import Sequence
import numpy as np
from pandas import read_excel
import SimpleITK as sitk
import matplotlib.pyplot as plt
import dicom



class Image(np.ndarray):

    def __new__(cls, image, origin, spacing, direction, is_dicom=None, rescaled=False):        
        obj = np.asarray(image).view(cls)
        obj.origin = origin
        obj.spacing = spacing
        obj.direction = direction
        obj.rescaled = rescaled
        obj.is_dicom = is_dicom
        return obj


    def __array_finalize__(self, obj):
        if obj is None: return
        self.origin = getattr(obj, 'origin', None)
        self.spacing = getattr(obj, 'spacing', None)
        self.direction = getattr(obj, 'direction', None)
        self.rescaled = getattr(obj, 'rescaled', False)
        self.is_dicom = getattr(obj, 'is_dicom', None)


    def to_itk(self):
        image = sitk.GetImageFromArray(self)
        image.SetOrigin(*self.origin)
        image.SetSpacing(*self.spacing)
        image.SetDirection(*self.direction)
        return image
    

    def to_vtk(self):
        return None
    

    @classmethod
    def from_itk(cls, dicom_dir_or_file_path):
        if os.path.isdir(dicom_dir_or_file_path):
            dicom_dir = dicom_dir_or_file_path
            image = dicom.read(dicom_dir)
            image.is_dicom=True

        else:
            image_file = dicom_dir_or_file_path
            try:
                sitk.ReadImage(image_file, imageIO='GDCMImageIO')
            except:
                image = sitk.ReadImage(image_file)
                image.is_dicom=False
            else:
                image = dicom.read(image_file)
                image.is_dicom=True
        
        image = Image(
                        sitk.GetArrayFromImage(image), 
                        np.array(image.GetOrigin()),
                        np.array(image.GetSpacing()),
                        np.array(image.GetDirection()),
                        is_dicom=image.is_dicom,
                        rescaled=False,
                    )

        return image
    

    @classmethod
    def from_numpy(cls, npypath):
        return np.load(npypath)


class Mask(np.ndarray):

    def __new__(cls, mask, origin, spacing, direction, label, automated=False, edited=False, lower_threshold=None, upper_threshold=None):        
        obj = np.asarray(mask).view(cls)
        obj.origin = origin
        obj.spacing = spacing
        obj.direction = direction
        obj.label = label
        obj.automated = automated
        obj.edited = edited
        obj.lower_threshold = lower_threshold
        obj.upper_threshold = upper_threshold
        return obj


    def __array_finalize__(self, obj):
        if obj is None: return
        self.origin = getattr(obj, 'origin', None)
        self.spacing = getattr(obj, 'spacing', None)
        self.direction = getattr(obj, 'direction', None)
        self.label = getattr(obj, 'label', None)
        self.automated = getattr(obj, 'automated', None)
        self.edited = getattr(obj, 'edited', False)
        self.lower_threshold = getattr(obj, 'lower_threshold', None)
        self.upper_threshold = getattr(obj, 'upper_threshold', None)


    def to_itk(self):
        image = sitk.GetImageFromArray(self)
        image.SetOrigin(*self.origin)
        image.SetSpacing(*self.spacing)
        image.SetDirection(*self.direction)
        return image
    

    def to_vtk(self):
        return None
    

    @classmethod
    def from_itk(cls, dicom_dir_or_file_path, label, automated):
        # returns a collection of masks
        if os.path.isdir(dicom_dir_or_file_path):
            dicom_dir = dicom_dir_or_file_path
            mask = Mask.from_dicom(dicom_dir)

        elif os.path.isfile(dicom_dir_or_file_path):
            image_file = dicom_dir_or_file_path
            mask = Mask.from_image(image_file)

        arr = sitk.GetArrayFromImage(mask)
        values = np.unique(arr)
        values = values[values!=0] # each value is a mask 
        labels = [label] * values.size # name/key of each mask
        masks = []
        if values.size > 1:
            for i,l in enumerate(labels):
                labels[i] = l+f'_{values[i]}'
        for v,l in zip(values,labels):
            masks.append(
                        Mask(
                            arr==v, 
                            np.array(mask.GetOrigin()),
                            np.array(mask.GetSpacing()),
                            np.array(mask.GetDirection()),
                            label=l,
                            automated=automated,
                            edited=False,
                            )
                        )
            
        return masks


    @classmethod
    def from_numpy(cls, npypath):
        return np.load(npypath)


class Landmark(np.ndarray): 

    dtype = np.dtype([('label', str, 12), ('coordinates', float, (3,))])

    def __new__(cls, name_coord_pairs=None, dtype=None):
        if dtype is None:
            dtype = cls.dtype
        
        if not name_coord_pairs:
            name_coord_pairs = []

        if isinstance(name_coord_pairs, dict):
            return np.array(list(name_coord_pairs.items()), dtype=dtype).view(cls)
        else:
            # isinstance(name_coord_pairs, Sequence|zip ):
            return np.array(list(name_coord_pairs), dtype=dtype).view(cls)


    def items(self):
        return self    


    def mutable(self):
        return {k:list(v) for k,v in self.items()}
    
    # convenience class
    def bilateral(self):
        return { n:v for n,v in self.items() if ('-L' in n or '-R' in n) and n !='Stm-L' }

    def left(self):
        return { n:v for n,v in self.items() if '-L' in n and n !='Stm-L' }

    def right(self):
        return { n:v for n,v in self.items() if '-R' in n}

    def midline(self):
        return { n:v for n,v in self.items() if '-L' not in n and '-R' not in n or n =='Stm-L'}

    def detached(self):
        return { n:v for n,v in self.items() if n in { "S", "U0R", "U1R-R", "U1R-L", "L0R", "L1R-R", "L1R-L", "COR-R", "COR-L" }}

    def computed(self):
        return { n:v for n,v in self.items() if "'" in n}
            
    def post(self):
        return { n:v for n,v in self.items() if n.endswith('--post')}


    @classmethod
    def from_text_file(cls, file):
        
        # this reads most of the common textual files containing 3D landmarks
        # uses one-based index if not landmark name is detected
        # returns a dictionary

        with open(file, 'r') as f:
            lines = f.read()

        # the following `if` is for CASS format
        if '\n' not in lines and ';' in lines:
            lines = lines.replace(';','\n')

        # for any uncommon line-break characters, follow the above approach:
        # do the replacement first, then match with regular expression
        # the line anchoring serves the following purpose
        # when automatically detecting labels, we want to match the entire line
        # thus anchoring the beginning of the line

        return cls(cls._reg_match(lines))


    @classmethod
    def from_excel(cls, file):
        lmk_pd = read_excel(file)
        
        coord_fields = ('Original_X', 'Original_Y', 'Original_Z')
        data_fields = ['Landmark Name',*coord_fields]
        if all([x in lmk_pd for x in data_fields]):
            # this file is probably from AA
            lmk_aa = lmk_pd[data_fields]
            if 'Landmark_Offset' in lmk_aa['Landmark Name'].values:
                offset_row_num = np.nonzero(lmk_aa['Landmark Name']=='Landmark_Offset')[0]
                assert(len(offset_row_num)==1)
                lmk_pd = lmk_aa.drop(index=offset_row_num)
                lmk_pd.loc[:,coord_fields] += lmk_aa.loc[offset_row_num,coord_fields].values

        labels, *coords = zip(*lmk_pd.dropna(axis=0).values)
        return cls(zip(labels, zip(*coords)))


    def write_csv(self, file):
        with open(file, 'w', newline='') as f:
                
            writer = csv.writer(f)
            for k,v in self.items():
                writer.writerow([k,*v])
            return None


    @staticmethod
    def _reg_match(input_string:str, sep:str=None, line_break:str=None):

        if not sep:
            sep = r'[ \t,]' #there must be either one comma or one horizontal white space
        if not line_break:
            line_break = r'[\n;]'

        # reg exp elements
        number = r'([+-]?\d+(?:\.\d+)?(?:[Ee]{1}[+-]?\d+)?)' # a full form: (+)(0)(.12)(E34)
        name = r'(\w+\'?(?:-[LRU])?(?:[_-]+\w+)?)'
        sep = rf'(?:[ \t]*{sep}[ \t]*)' 
        line_break = rf'[ \t]*{line_break}' # takes care of empty lines too

        # reg exp match each line
        reg_line = rf'^[ \t]*(?:{name}{sep})?{number}{sep}{number}{sep}{number}(?:{line_break})?' # anchor the line begining but not the end
        matches = re.findall(reg_line, input_string, re.MULTILINE|re.DOTALL)
        if not len(matches):
            return {}
        labels, *coords = zip(*matches)
        labels = [l if l else str(i+1) for i,l in enumerate(labels)]
        coords = [(float(x),float(y),float(z)) for x,y,z in zip(*coords)]
        return zip(labels, coords)


class PatientData:
    def __init__(self) -> None:
        pass


    @property
    def image(self):
        if hasattr(self, '_image'):
            return self._image
        return None
    

    @image.setter
    def image(self, image):
        if hasattr(self, '_image'):
            raise ValueError('image already exists')
        self._image = image


    @property
    def mask(self):
        if self.image is None:
            return None
        if not hasattr(self, '_mask'):
            self._mask = []
        return self._mask
        
    
    @property
    def landmark(self):
        if self.image is None:
            return None
        if not hasattr(self, '_landmark'):
            self._landmark = Landmark()
        return self._landmark
    

    @landmark.setter
    def landmark(self, ldmk):
        self._landmark = ldmk



    def import_image(self, image_path):
        self.image = Image.from_itk(image_path)
        self.image.flags.writeable = False

        return None


    def import_mask(self, mask_path, mask_label, automated):
        # label must not contain PHI

        if self.image is None:
            print('no image')
            return None
        
        masks = Mask.from_itk(mask_path, mask_label, automated=automated)

        for _ in masks:
            self.mask.append(_)

        return None


    def import_landmark(self, ldmk_path, automated, override=True):

        if self.image is None:
            print('no image')
            return None
        if ldmk_path.endswith('.xlsx'):
            ldmk = Landmark.from_excel(ldmk_path)
        else:
            ldmk = Landmark.from_text_file(ldmk_path)

        for i,(k,v) in enumerate(ldmk.items()):
            if k in self.landmark['label'] and not override:
                continue
            self.landmark = np.append(self.landmark, ldmk[i]) # not efficient

        return None


    def load_image(self, npy_file):
        self.image = Image.from_numpy(npy_file)


    def load_mask(self, npy_file):
        self.mask.append(Mask.from_numpy(npy_file))


    def create_mask_with_threshold(self, lower_threshold, upper_threshold, label):
        # thresholds are half-inclusive
        if self.image.is_dicom and not self.image.rescaled:
            raise ValueError('must rescale dicom image first before applying threshold')
        return Mask(
                    np.logical_and(self.image>=lower_threshold, self.image<upper_threshold),
                    self.image.origin,
                    self.image.spacing,
                    self.image.direction,
                    label=label,
                    automated=False,
                    edited=False,
                    lower_threshold=lower_threshold,
                    upper_threshold=upper_threshold
                    )


    def test_show(self):
        slice = 100
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot()
        image_slice = self.image[slice,:,:]
        ax.imshow(image_slice, cmap=plt.cm.gray)
        masks_color = np.zeros(image_slice.shape)
        for i in range(len(self.mask)):
            masks_color[self.mask[i][slice,:,:]] = i+1
        masks_color[masks_color==0] = np.nan
        ax.imshow(masks_color, alpha=.5)
        lmk = self.landmark['coordinates']
        ax.scatter(lmk[:,0],lmk[:,1])
        plt.show()

