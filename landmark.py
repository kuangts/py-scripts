# rewrite with this:
# https://kitware.github.io/vtk-examples/site/Cxx/Interaction/MoveAGlyph/

import re
import sqlite3
from typing import Any
from math import isnan
from copy import deepcopy
from collections import namedtuple
from collections.abc import Sequence, Iterable
from operator import itemgetter, attrgetter

import numpy as np
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

default_db_location = r'C:\py-scripts\CASS.db'
landmark_entry_fields = ('ID','M','Category','Group','Name','Fullname','Description','Coordinate','View','DP') # corresponding to the db
soft_tissue_labels_23 = {
                            "Gb'", 'Ls', 'Sl', 'En-R', 'Stm-L', "Zy'-L", 'Stm-U', "Go'-L", 'Ex-R', 'Sn',
                            'Prn', "Pog'", 'Ex-L', "Go'-R", 'En-L', 'Li', 'CM', "N'", 'Ch-R', "Me'",
                            'C', "Zy'-R", 'Ch-L'
                        }

LDMK = namedtuple('LDMK', landmark_entry_fields)


class Library(frozenset): # immutable set containing immutable landmark entries
    """
    immutable set containing immutable landmark definition
    this class is to bridge landmark database (CASS.db) and landmark manipulation in python.
    it provides simple functions for look-up and filter operations.
    this is a subclass of Set/forzenset and therefore it is, again
    IMMUTABLE and UNORDERED
    """
    @classmethod
    def from_db(cls, location):
        if not hasattr(cls, 'lib'):
            cls.lib = {}
        if not location in cls.lib:
            cls.lib[location] = cls(location)
        return cls.lib[location]

    def to_dict(self):
        return LandmarkDict({x.Name:x.Coordinate for x in self.sorted()})

    def __new__(cls, db_location_or_existing_set=default_db_location):
        if isinstance(db_location_or_existing_set, str): 
            with sqlite3.connect(db_location_or_existing_set) as con:
                # select all from table Landmark_Library
                # unpack the resulting cursor object
                # create item with each row of the table
                # reference fields of each item with dot notation or tuple indexing
                lib = con.execute(f'SELECT * FROM Landmark_Library').fetchall()
                cat = con.execute(f'SELECT * FROM Measure_Items').fetchall()
                ana = con.execute(f'SELECT * FROM Analysis_Name').fetchall()

            # order and names of categories
            cat_names = {x[1]:x[2] for x in ana}
            category = {}

            # order and landmarks within each category
            for x in cat:
                cat_id = cat_names[x[2]]
                m_order = int(x[0])
                if cat_id not in category:
                    category[cat_id] = {}
                category[cat_id][m_order] = x[5]
            del category['New GT'] # because m_order has duplicates

            # after assuring each landmark (lmk) belongs to one and only one group (grp)
            lmk_group = {lmk:grp for grp, mem in category.items() for lmk in mem.values() if lmk }

            # following items are the content of the library
            items = [
                LDMK(
                    ID=int(x[0]),
                    M=int(x[1]),
                    Category=x[2],
                    Group=lmk_group[x[4]],
                    Name=x[4],
                    Fullname=x[5],
                    Description=x[6],
                    Coordinate=(float(x[7]),float(x[8]),float(x[9])),
                    View=x[10],
                    DP=x[11],
                )
                for x in lib if x[4] in lmk_group # use only known landmarks
            ]
        else: # creating library from existing set, likely a subset of the full library
            items = db_location_or_existing_set
            assert all(isinstance(x, LDMK) for x in items), f'cannot create library with input {db_location_or_existing_set}'

        obj = super().__new__(cls, items)
        return obj

    def __copy__(self):
        return self

    def __contains__(self, __name):
        if isinstance(__name, str):
            return __name in self.field('Name')
        return super().__contains__(__name)

    def field(self, fieldname):
        # extracts field or fields into set
        # fieldname is given in a string
        # the returned set has specified field of the namedtuple (LDMK)
        if isinstance(fieldname, str):
            if fieldname not in LDMK._fields:
                raise ValueError(f'{fieldname} is not a valid field')
            return { getattr(x, fieldname) for x in self }
        else:
            raise ValueError('fieldname is not valid input')
    
    def split(self, fieldnames):
        # extracts fields into lists
        for f in fieldnames:
            if f not in LDMK._fields:
                raise ValueError(f'{f} is not a valid field')
        return tuple(zip(*[[getattr(x, f) for f in fieldnames] for x in self ]))
    
    def sorted(self, key_or_fieldname='Group', reverse=False):
        key = attrgetter(key_or_fieldname) if isinstance(key_or_fieldname, str) else key_or_fieldname
        return sorted(self, key=key, reverse=reverse)

    def find(self, **kwargs):
        # finds the first match of LDMK instance
        # or returns None is none is found
        # use **kwargs to specify matching condition
        # e.g. library.find(Name='Fz-R') will find LDMK x where x.Name=='Fz-R'
        # e.g. library.find(Category='Skeletal') will find and return only the first skeletal landmark
        # to find all matches, use filter instead: e.g. library.filter(lambda x: x.Category=='Skeletal')
        for x in self:
            if all(map(lambda a:getattr(x,a[0])==a[1], kwargs.items())):
                return x
        return None

    def filter(self, callable_on_lib_entry=lambda x:x.Name!=""):
        # returns subset of items matching criteria
        # similar to in-built filter function
        # takes a lambda which is applied onto library entries
        # e.g. library.filter(lambda x: 'Fz' in x.Name) will find Fz-R and Fz-L
        return self.__class__({x for x in self if callable_on_lib_entry(x)})

    # following logical operators only consider Name of each element
    def isdisjoint(self, other: Iterable[Any]) -> bool:
        return self.field('Name').isdisjoint(other.field('Name'))

    def union(self, other):
        if not self.isdisjoint(other):
            raise ValueError('landmark sets performing union must be disjoint')
        return self.__class__(super().union(other))

    def difference(self, other):
        return self.filter(lambda x: x.Name in 
            self.field('Name').difference(other.field('Name'))
        )

    def intersection(self, other):
        return self.filter(lambda x: x.Name in 
            self.field('Name').intersection(other.field('Name'))
        )

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __and__(self, other):
        return self.intersection(other)

    # convenience filter methods below
    def group(self, grp):
        return self.filter(lambda x:x.Group==grp)

    def category(self, cat):
        return self.filter(lambda x:x.Category==cat)

    def bilateral(self):
        return self.filter(lambda x: ('-L' in x.Name or '-R' in x.Name) and x.Name !='Stm-L')

    def left(self):
        return self.filter(lambda x: '-L' in x.Name and x.Name !='Stm-L')

    def right(self):
        return self.filter(lambda x: '-R' in x.Name)

    def midline(self):
        return self.filter(lambda x: '-L' not in x.Name and '-R' not in x.Name or x.Name =='Stm-L')

    def detached(self):
        return self.filter(lambda x: x.Name in { "S", "U0R", "U1R-R", "U1R-L", "L0R", "L1R-R", "L1R-L", "COR-R", "COR-L" })

    def computed(self):
        return self.filter(lambda x: "'" in x.Name)

    def soft_tissue_23(self):
        return self.filter(lambda x: x.Name in soft_tissue_labels_23)
        
    def skeletal(self):
        return self.filter(lambda x: x.Category == 'Skeletal')
        
    def soft_tissue(self):
        return self.filter(lambda x: x.Category == 'Soft Tissue')
        
    def string(self, sort_key=attrgetter('ID'), group_key=attrgetter('Group'), indent=' ', **passthrough):
        if sort_key is not None:
            new_coll = self.sorted(sort_key)
        else:
            new_coll = self
        if group_key is not None:
            str_dict = {}
            sub_indent = indent+'  '
            for x in new_coll:
                g = str(group_key(x))
                if g not in str_dict:
                    str_dict[g] = []
                str_dict[g].append(x.string(group=False, indent=sub_indent, **passthrough))
            return_str = f'\n\n{indent}'.join(f'\n{sub_indent}'.join([k+':',*str_dict[k]]) for k in sorted(str_dict.keys()))
        else:
            return_str = f'\n{indent}'.join(x.string(indent=indent, **passthrough) for x in new_coll)
        return indent + return_str + '\n'
    
    def __repr__(self):
        return self.string(truncate_definition=False, coordinate=False)

class LandmarkSet(Library):
    """
    immutable set containing immutable landmark entries including coordinates
    this class uses the same structure as `Library`
    but enables landmark reading.
    IMMUTABLE and UNORDERED
    """

    def __new__(cls, file_location_or_existing_set, db_location=default_db_location, **parseargs):
        # read the db to get landmark definitions
        db = super().__new__(cls, db_location)
        if isinstance(file_location_or_existing_set, str):
            if not file_location_or_existing_set:
                coord_dict = {}
            else:
                # read landmark string
                with open(file_location_or_existing_set, 'r') as f:
                    lmk_str = f.read()
                # parse landmark string
                labels, coordinates, _ = LandmarkDict.parse(lmk_str, **parseargs)
                coord_dict = dict(zip(labels, coordinates))
                # check if there is unknown landmark
                if not all(x in db.field('Name') for x in coord_dict):
                    print('WARNING: some labels are not recognized')
            # prepare for instantiation
            lmk_set = {x._replace(Coordinate=tuple(coord_dict[x.Name])) for x in db}
        else:
            lmk_set = file_location_or_existing_set

        # create landmark set
        obj = super().__new__(cls, lmk_set)

        return obj
            

    def __getitem__(self, __key):
        if isinstance(__key, str):
            x = self.find(Name=__key)
            if x is None:
                return (float('nan'),)*3
            return x.Coordinate
        elif isinstance(__key, Sequence) and all(isinstance(x,str) for x in __key):
            return [self.__getitem__(k) for k in __key]
        else:
            raise ValueError('cannot get item: wrong key')
            
    def set_coordinates(self, __key, __value):
        if isinstance(__key , str):
            return self.set_coordinates([__key],[__value])
        elif isinstance(__key, Sequence) and isinstance(__value, Sequence) and len(__key) == len(__value) and all(isinstance(x, str) for x in __key) and all(len(v)==3 for v in __value):
            return self.__class__({x._replace(Coordinate=tuple(__value[__key.index(x.Name)])) if x.Name in __key else x for x in self})
        else:
            raise ValueError('cannot set coordinates')

    def __setitem__(self, __key, __value) -> None:
        raise ValueError('setting item not allowed for immutable set')

    def __delitem__(self, __key) -> None:
        raise ValueError('deleting item not allowed for immutable set')

    @property
    def centroid(self):
        return np.asarray(list(self.field('Coordinate'))).mean(axis=0).tolist()

    def translate(self, t):
        labels, coords = self.split(('Name','Coordinate'))
        coords_dict = dict(zip(labels, (np.asarray(coords) + t).tolist()))
        return self.__class__({ x._replace(Coordinate=tuple(coords_dict[x.Name])) for x in self })

    def transform(self, T, post_multiply=True): 
        T = np.array(T)
        if T.shape not in ((3,3),(4,4)):
            raise ValueError(f'incompatible matrix shape: {T.shape} ')
        labels, coords = self.split(('Name','Coordinate'))
        if T.shape == (4,4):
            coords = np.hstack((coords, np.ones((len(coords),1))))
        new_coords = coords @ T if post_multiply else coords @ T.T
        coords_dict = dict(zip(labels, new_coords[:,:3].tolist()))
        return self.__class__({ x._replace(Coordinate=tuple(coords_dict[x.Name])) for x in self })

    def __repr__(self):
        return self.string(sort_key=attrgetter('ID'), group_key=None, group=False, id=False, definition=False)


class LDMK(LDMK):

    def print_str(self):
        d = dict(zip(landmark_entry_fields, (
            f'[{self.ID:>3}]',
            f'{self.M}',
            f'{self.Category}',
            f'{self.Group[3:]:<12}',
            f'{self.Name:<10}',
            f'{self.Fullname.strip()}',
            f'{self.Description.strip()}',
            f'({self.Coordinate[0]:7.2f}, {self.Coordinate[1]:7.2f}, {self.Coordinate[2]:7.2f})', # len=27
            f'{self.View}',
            f'{self.DP}',
        )))
        d['Name_Group'] = f'{self.Name} <{self.Group[3:]}>' + ' '*max(23 - len(self.Name) - len(self.Group),0) # len=20
        d['Definition'] = '{Fullname}: {Description}'.format(**d)
        d['Definition_truncated'] = d['Definition']
        if len(d['Definition']) > 80:
            d['Definition_truncated'] = d['Definition'][:75] + ' ... '
        return d

    def string(self, id=True, group=True, coordinate=True, definition=True, truncate_definition=True, indent=''):
        print_str = self.print_str()
        fstr = ''
        if id:
            fstr += '{ID} '
            indent += ' '*(len(print_str['ID'])+1)
        fstr += '{Name_Group}' if group else '{Name}'
        if coordinate:
            fstr += '{Coordinate}'
            if definition:
                fstr += f'\n{indent}' + ('{Definition_truncated}' if truncate_definition else '{Definition}')
        elif definition:
            fstr += '{Fullname}' + f'\n{indent}' + '{Description}'

        return fstr.format(**print_str)

    def __repr__(self):
        return self.string()

        
class LandmarkDict(dict):
    '''
    stores label:coordinate
    '''

    @classmethod
    def from_text(cls, file, **parseargs):
        # read text file, mainly txt and csv
        with open(file, 'r') as f:
            lmk_str = f.read()
            labels, coordinates, header = cls.parse(lmk_str, **parseargs)
            obj = cls(zip(labels, coordinates))
            setattr(obj, 'header', header)
            return obj

    @classmethod
    def from_excel(cls, file):
        # read excel sheet with very specific format - exported from CASS
        # use from_cass instead if possible, and if db version is not an issue
        import pandas
        V = pandas.read_excel(file, header=0, engine='openpyxl').values
        return cls(zip(V[1:,0], (V[1:,1:4]+V[0,1:4]).tolist()))

    @classmethod
    def from_cass(cls, file, interpret=True):
        import rarfile
        with rarfile.RarFile(file) as f:
            with f.open('Measure_Data_Info_new.bin') as l:
                t = l.read().decode("utf-8").strip(';').split(';')
                lmk = [x.split(',') for x in t]
                lmk = {int(l[0]):tuple(map(float,l[-3:])) for l in lmk}
        
        if interpret:
            lmk_new = {}
            lmk_del = {}
            for k,v in lmk.items():
                x = Library.from_db(default_db_location).find(ID=k)
                if x is not None:
                    lmk_new[x.Name] = v
                else:
                    lmk_del[k] = v
            lmk = lmk_new

        return cls(lmk)


    def copy(self):
        return deepcopy(self)

    def __getitem__(self, __key):
        if __key in self:
            val = super().__getitem__(__key)
        else:
            val = [float('nan')]*3
        return val

    def __setitem__(self, __key, __value) -> None:
        if isinstance(__value, np.ndarray):
            __value = __value.tolist()
        if any(map(isnan,__value)):
            print('WARNNING: coordinates must have finite value')
            return None
        return super().__setitem__(__key, __value)

    def __delitem__(self, __key) -> None:
        if __key not in self:
            print(f'WARNING: landmark {__key} is not present')
            return None
        return super().__delitem__(__key)

    def coordinates(self, ordered_labels=None):
        new_self = self.copy()
        if ordered_labels is None:
            ordered_labels = self.keys()
        return [new_self[k] for k in ordered_labels]

    def select(self, ordered_labels):
        new_self = self.copy()
        return_self = self.__class__()
        result = self.__class__({k:new_self[k] for k in ordered_labels})
        return result

    def set_coordinates(self, coords):
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        for k,c in zip(self.keys(), coords):
            self[k] = list(c)
        return None

    def __add__(self, t): # translation
        new_self = self.copy()
        if type(t) == type(self):
            new_self.update(t)
            return new_self
        for k in new_self.keys():
            for i in range(3):
                new_self[k][i] += +t[i]
        return new_self

    def __sub__(self, t): # translation
        new_self = self.copy()
        if type(t) == type(self):
            #############################################################################################
            new_self.update(t)
            return new_self
        for k in new_self.keys():
            for i in range(3):
                new_self[k][i] -= t[i]
        return new_self

    def __matmul__(self, T): # rotation
        T = np.asarray(T)
        if T.shape not in ((3,3),(4,4)):
            raise ValueError(f'cannot matmul with shape {T.shape}')
        new_self = self.copy()
        coords = np.asarray(self.coordinates())
        if T.shape == (4,4):
            coords = np.hstack((coords, np.ones((coords.shape[0],1))))
        coords = coords @ T
        new_self.set_coordinates( coords[:,:3] )
        return new_self

    def write(self, file, **kwargs):
        write_str = self.string(**kwargs)
        if write_str:
            with open(file,'w') as f:
                f.write(write_str)


    @staticmethod
    def parse(lmk_str, num_lines_header=None, separator='[,: ]+', line_break='[;\n]+', nan_str={'N/A','n/a','NA','na','nan','NaN'}):        
        '''
        `num_lines_header` stores the number of lines before data begins. it must be set if not `None` - automatic determination is not supported yet.
        checks any combination of ';' and '\n' for line breaks
        checks any combination of ',', ':', and space for delimiter within each line
        detects if label is present
            with labels, each line starting with alphabetic characters
            without label, numeric only, use 0-based index in string as label
        '''
        labels, coordinates, header = [],[],[]

        # 'header' stores number of lines in the begining of file before data, might be different from pandas, performs n splits where n equals 'header'
        if num_lines_header:
            lmk_str = lmk_str.split('\n', num_lines_header) 
            header, lmk_str = lmk_str[0:num_lines_header], lmk_str[-1]

            # regexp checks for patterns ';\n' or '\n' or ';' (in order given by a 'Sequence' of str), then splits at all occurrences 

        assert line_break and separator, 'must provide pattern for parsing'

        lmk_str = lmk_str.strip().strip(line_break) # removes trailing newlines and semicolon to prevent creation of '' after split
        lines = re.split(line_break, lmk_str)
        for i,line in enumerate(lines):
            if re.fullmatch('^[\s]*[a-zA-Z]+.*', line) is not None:
                # label is present - read four columns
                label, *coord = re.split(separator, line.strip())
            else:
                coord = re.split(separator, line.strip())
                if len(coord) == 4:
                    label, *coord = coord
                elif len(coord) == 3:
                    label = str(i)

            assert len(coord)==3, 'split went wrong'

            if any([ x.strip() in nan_str  for x in coord]): # ignore nan's
                continue

            coord = [float(x) for x in coord] 

            if not any(coord): # all zero scenario, legacy reason
                continue

            labels.append(label.strip())
            coordinates.append(coord)

        return labels, coordinates, header

    def string(self, ordered_label=None, formatted=False, keep_label=True, keep_coordinate=True, header=None, separator=',', line_break='\n'):
        '''
        `header` specifies the header content of the coming file, in a `list` or `str`. a `list` of three `str`s would correspond to three header lines. the default is `None`, no header. if set to `''`, it will try to write header using previous header if possible.
        nan values are written to file as nan
        '''

        # line_break='\n' -> csv
        # line_break=';'  -> cass-readable, not recommended -- read from cass file directly if possible, or if cass is saved with the latest db file.
        # line_break=None -> do not join lines - returns list for further processing

        # header string
        if header is None:
            header = ''
        else:
            header = header if header else self.header
            header = '\n'.join(header if isinstance(header, Sequence) and not isinstance(header,str) else [header]) + '\n'
        
        # landmark string
        lmk = ''
        if formatted:
            if len(self):
                header += '   LABEL  |        X        Y        Z\n' + '-'*40 + '\n'
                lmk = '\n'.join([f'{l:10}| {x[0]:8.3f} {x[1]:8.3f} {x[2]:8.3f}' for l,x in self.items()])+'\n\n'
            lmk += f'total: {len(self)}\n'
        else:
            lmk = []
            if ordered_label is None:
                ordered_label = self.keys() 
            for label in ordered_label:
                coord = self[label]
                l = []
                if keep_label:
                    l += [label]
                if keep_coordinate:
                    l += [*map(str,coord)]
                lmk += [l]

            if separator is None:
                return lmk
            lmk = [separator.join(l).strip(separator) for l in lmk]

            if line_break is None:
                return lmk
            lmk = line_break.join(lmk).strip(line_break)

            if lmk:
                lmk += line_break
        
        # combine
        return header + lmk


    @property
    def header(self):
        # n items on the list _header represent n lines of header
        if not hasattr(self, '_header'):
            setattr(self, '_header', ())
        return self._header 

    @header.setter
    def header(self, value):
        if isinstance(value, str):
            value = (value,)
        setattr(self, '_header', value)


    def move_to_mask(self, mask, threshold=None):

        import pkg_resources
        try:
            pkg_resources.require(['SimpleITK','scipy','numpy'])
        except Exception as e:
            print(e)
            return None
        import SimpleITK as sitk
        from scipy.ndimage import binary_dilation, binary_erosion

        lmk_not_moved, lmk = self.select({'Detached'}, return_remaining=True)
        ind2coord = lambda index: np.array([ mask.TransformIndexToPhysicalPoint(ind.tolist()) for ind in index ])
        closest = lambda l, bd: bd[np.argmin(np.sum((bd - l)**2, axis=1)),:]

        if isinstance(mask, sitk.SimpleITK.Image):
            arr = sitk.GetArrayFromImage(mask)
        else:
            print('wrong input argument')
            return None 
        arr = arr>0
        
        # pass 1 - dilation
        arr1_bd = np.logical_xor(arr, binary_dilation(arr))
        arr1_bd_ind = np.array(np.nonzero(arr1_bd)).T[:,::-1]
        coords_bd1 = ind2coord(arr1_bd_ind)

        # pass 2 - erosion
        arr2_bd = np.logical_xor(arr, binary_erosion(arr))
        arr2_bd_ind = np.array(np.nonzero(arr2_bd)).T[:,::-1]
        coords_bd2 = ind2coord(arr2_bd_ind)

        # average two passes
        coords_new = np.array([ closest(l, coords_bd1)/2 + closest(l, coords_bd2)/2 for l in self.coordinates ])

        # check dist with thres
        if threshold!=None:
            d = np.sum((coords_new-self.coordinates)**2, axis=1)**.5
            ind = d>threshold
            coords_new[ind] = self.coordinates[ind]

        lmk = LandmarkDict(zip(lmk.keys(), coords_new))
        lmk.update(lmk_not_moved['Detached'])

        return lmk


class vtkLandmark:

    Color = (1,0,0)
    HighlightColor = (0,1,0)

    def __init__(self, label:str, coord=(float('nan'),)*3, **kwargs):
        src = vtkSphereSource()
        src.SetCenter(*coord)
        src.SetRadius(1)
        map = vtkPolyDataMapper()
        map.SetInputConnection(src.GetOutputPort())
        act = vtkActor()
        act.SetMapper(map)
        act.GetProperty().SetColor(*self.Color)
        txt = vtkBillboardTextActor3D()
        txt.SetPosition(*coord)
        txt.SetInput(label)
        txt.GetTextProperty().SetFontSize(24)
        txt.GetTextProperty().SetJustificationToCentered()
        txt.GetTextProperty().SetColor(*self.Color)
        txt.PickableOff()

        self.sphere = src
        self.sphere_actor = act
        self.label_actor = txt
        self.sphere_actor.prop_name = 'ldmk'
        self.sphere_actor.parent = self

        for k,v in kwargs.items():
            setattr(self, k, v)

    @property
    def label(self):
        return self.label_actor.GetInput().strip()

    def move_to(self, new_coord):
        self.sphere.SetCenter(*new_coord)
        self.sphere.Update()
        self.label_actor.SetPosition(*new_coord)

    def set_renderer(self, ren):
        self.renderer = ren
        ren.AddActor(self.sphere_actor)
        ren.AddActor(self.label_actor)

    def remove(self):
        self.renderer.RemoveActor(self.sphere_actor)
        self.renderer.RemoveActor(self.label_actor)
        del self.sphere_actor, self.label_actor, self.sphere

    def refresh(self):
        if hasattr(self, 'selected') and self.selected:
            self.sphere_actor.GetProperty().SetColor(*self.HighlightColor)
            self.label_actor.GetTextProperty().SetColor(*self.HighlightColor)
        else:
            self.sphere_actor.GetProperty().SetColor(*self.Color)
            self.label_actor.GetTextProperty().SetColor(*self.Color)

        self.sphere.Update()

    def select(self):
        self.selected = True
        self.refresh()

    def deselect(self):
        self.selected = False
        self.refresh()



if __name__=='__main__':

    lmk = LandmarkSet(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\extracted_from_cass.csv')
    # lmk = lmk.bilateral()
    # lmk.transform(np.eye(3))
    # lmk.transform(np.eye(4)).field('Coordinate')
    # lmka = lmk.set_coordinates('A',(1,2,3))
    # print(lmk['A'])
    # print(lmka['A'])
    # print(lib)
    print(len(lmk))
    print(lmk['Zy-R'])
#     parser = argparse.ArgumentParser(allow_abbrev=True)
#     parser.add_argument('--input', type=str, nargs='?')
#     parser.add_argument('--output', type=str, nargs='?')
#     parser.add_argument('--num-header', type=int, default=None)
#     parser.add_argument('--read-nan', nargs='+', default=('N/A','n/a','NA','na','nan','NaN')) 
#     parser.add_argument('--read-delimiter', type=str, default='[,: ]+') 
#     parser.add_argument('--read-line-break', type=str, default='[;\n]+')
#     parser.add_argument('--header', nargs='*', type=str)
#     parser.add_argument('-nan','--nan-string', type=str, default='')
#     parser.add_argument('-d','--delimiter', type=str, default=',')
#     parser.add_argument('-b','--line-break', type=str, default='\n')
#     parser.add_argument('--no-label', action='store_true', default=False) # print only coordinates
#     parser.add_argument('--label-only', action='store_true', default=False) # print only labels

#     args = parser.parse_args()

#     if args.input is not None:
#         if args.input.endswith('.xlsx'):
#             lines = Landmark.from_excel(args.input).string()
#         else:
#             with open(args.input, 'r') as f:
#                 lines = f.read()
#     else:
#         lines = ''.join([line for line in sys.stdin])

#     lmk = Landmark.parse(
#                 lines,
#                 header=args.num_header,
#                 nan_str=args.read_nan,
#                 separator=args.read_delimiter,
#                 line_break=args.read_line_break)
#     lmk_str = lmk.string(
#                 header=args.header,
#                 formatted=args.formatted,
#                 nan_str=args.nan_string,
#                 keep_label=not args.no_label,
#                 keep_coordinate=not args.label_only,
#                 separator=args.delimiter,
#                 line_break=args.line_break)

#     if args.output is not None:
#         with open(args.output, 'wt', newline='') as f:
#             f.write(lmk_str)
#     else:            
#         sys.stdout.write(lmk_str)

