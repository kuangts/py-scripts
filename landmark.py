import re, math, collections, sqlite3, operator
from vtkmodules.vtkFiltersSources import vtkSphereSource 
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

_nan = float('nan')
_anynan = lambda x: any(map(math.isnan,x))
db_location = r'C:\AA\AA.Release.GT\CASS.db'


class Library(frozenset):
    """
    IMMUTABLE and UNORDERED
    this class is to bridge landmark database (CASS.db) and landmark manipulation in python.
    it provides simple functions for look-up and filter operations.
    this is a subclass of forzenset and therefore it is
    IMMUTABLE and UNORDERED
    """

    # class for entries
    Entry = collections.namedtuple('LDMK_ENTRY',('ID','M','Category','Group','Name','Fullname','Description','Coordinate','View','DP'))
    setattr(Entry,'__repr__',lambda s: f'[{s.ID:>3}] {s.Name:<10}|{s.Group[3:]:^15}|{s.Category:^15}| {s.Fullname}')

    def __new__(cls, db_location_or_existing_set=db_location):
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
                cls.Entry(
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
            assert all(isinstance(x, cls.Entry) for x in items), f'cannot create library with input {db_location_or_existing_set}'

        obj = super().__new__(cls, items)
        return obj

    def __repr__(self):
        s = '\n'.join(x.__repr__() for x in self) + '\n\n'
        s += f'total {len(self)} landmarks in library'
        return s

    def field(self, fieldname):
        # extracts field or fields into list object
        # if fieldname is given in a string, then the returned list has members of self.Entry instances
        # if fieldname is given in a sequence, e.g. list, then each member is wrapped in a list as well
        if isinstance(fieldname, str):
            if fieldname not in self.Entry._fields:
                print(f'{fieldname} is invalid input')
                return {}
            return { getattr(x, fieldname) for x in self }
        elif isinstance(fieldname, collections.abc.Sequence):
            for f in fieldname:
                if f not in self.Entry._fields:
                    print(f'{f} is invalid input')
                    return {}
            return { [getattr(x, f) for f in fieldname] for x in self }
    
    def find(self, **kwargs):
        # finds the first match of self.Entry instance
        # or returns None is none is found
        # use **kwargs to specify matching condition
        # e.g. library.find(Name='Fz-R') will find entry x where x.Name=='Fz-R'
        # e.g. library.find(Category='Skeletal') will find and return only the first skeletal landmark
        # to find all matches, use filter instead: e.g. library.filter(lambda x: x.Category=='Skeletal')
        for x in self:
            if all(map(lambda a:getattr(x,a[0])==a[1], kwargs.items())):
                return x

    def filter(self, callable_on_lib_entry=lambda x:x.Name!=""):
        # returns subset of items matching criteria
        # similar to in-built filter function
        # takes a lambda which is applied onto library entries
        # e.g. library.filter(lambda x: 'Fz' in x.Name) will find Fz-R and Fz-L
        return self.__class__({x for x in self if callable_on_lib_entry(x)})


    # operator functions
    def union(self, other):
        return self.__class__(super().union(other))

    def difference(self, other):
        return self.__class__(super().difference(other))

    def intersection(self, other):
        return self.__class__(super().intersection(other))

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

    def jungwook(self):
        return self.filter(lambda x: x.Name in 
            {
                "Gb'",
                "N'",
                "Zy'-R",
                "Zy'-L",
                "Pog'",
                "Me'",
                "Go'-R",
                "Go'-L",
                "En-R",
                "En-L",
                "Ex-R",
                "Ex-L",
                "Prn",
                "Sn",
                "CM",
                "Ls",
                "Stm-U",
                "Stm-L",
                "Ch-R",
                "Ch-L",
                "Li",
                "Sl",
                "C",
            }
        )


class LandmarkDict(dict):
    '''
    this is a convenience class for handling landmark files of various formats
    use exactly like a dictionary
    be mindful of the data type put in - list and numpy.ndarray both work with the class
    '''

    @classmethod
    def read(cls, file, **parseargs):
        # read text file, mainly txt and csv
        with open(file, 'r') as f:
            lmk_str = f.read()
            return cls.parse(lmk_str, **parseargs)

    @classmethod
    def from_excel(cls, file):
        # read excel sheet with very specific format
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
                x = library.find(ID=k)
                if x is not None:
                    lmk_new[x.Name] = v
                else:
                    lmk_del[k] = v
            lmk = lmk_new

        return cls(lmk)

    def write(self, file, **kwargs):
        write_str = self.string(**kwargs)
        if write_str:
            with open(file,'w') as f:
                f.write(write_str)

    def remove_nan(self):
        for l in list(self.keys()):
            if _anynan(self[l]):
                self.pop(l)
        return self

    @property
    def coordinates(self):
        return list(self.values())

    @property
    def labels(self):
        return list(self.keys())

    @classmethod
    def parse(cls, lmk_str, num_lines_header=None, separator='[,: ]+', line_ending='[;\n]+', nan_str={'N/A','n/a','NA','na','nan','NaN'}):        
        '''
        `num_lines_header` stores the number of lines before data begins. it must be set if not `None` - automatic determination is not supported yet.
        checks any combination of ';' and '\n' for line breaks
        checks any combination of ',', ':', and space for delimiter within each line
        detects if label is present
            with labels, each line starting with alphabetic characters
            without label, numeric only, use 0-based index in string as label
        '''
        lmk = cls()

        # 'header' stores number of lines in the begining of file before data, might be different from pandas, performs n splits where n equals 'header'
        if num_lines_header:
            lmk_str = lmk_str.split('\n', num_lines_header) 
            lmk.header, lmk_str = lmk_str[0:num_lines_header], lmk_str[-1]

            # regexp checks for patterns ';\n' or '\n' or ';' (in order given by a 'Sequence' of str), then splits at all occurrences 

        assert line_ending and separator, 'must provide pattern for parsing'

        lmk_str = lmk_str.strip().strip(line_ending) # removes trailing newlines and semicolon to prevent creation of '' after split
        lines = re.split(line_ending, lmk_str)
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

            coord = [_nan if x.strip() in nan_str else float(x) for x in coord] # if nan, assign nan to coordinate 

            if not any(coord): # all zero scenario
                coord = [_nan]*3 
            lmk[label.strip()] = coord
        return lmk

    def string(self, nan_str='0.0', formatted=False, keep_label=True, keep_coordinate=True, header=None, separator=',', line_ending='\n'):
        '''
        `header` specifies the header content of the coming file, in a `list` or `str`. a `list` of three `str`s would correspond to three header lines. the default is `None`, no header. if set to `''`, it will try to write header using previous header if possible.
        an nan value is written to file as `nan_str`. if `nan_str` is set to '0.0', then nan coordinates are written as '0.0, 0.0, 0.0', which is the default
        if nan_str is '', all entries containing nan value(s) are removed
        '''

        # line_ending='\n' -> csv
        # line_ending=';'  -> cass-readable, not recommended -- read from cass file directly if possible, or if cass is saved with the latest db file.
        # line_ending=None -> do not join lines - returns list for further processing

        lmk = []

        if header is None:
            header = ''
        else:
            header = header if header else self.header
            header = '\n'.join(header if isinstance(header, collections.abc.Sequence) and not isinstance(header,str) else [header]) + '\n'
        
        remove_nan = len(nan_str)==0

        if formatted:
            if not self.len(remove_nan=remove_nan):
                lmk = ''
            else:
                header += '   LABEL  |        X        Y        Z\n' + '-'*40 + '\n'
                lmk = '\n'.join([f'{l:10}| {x[0]:8.3f} {x[1]:8.3f} {x[2]:8.3f}' for l,x in self.items()])+'\n\n'
                if self.len(remove_nan=False)==len(self):
                    lmk += f'Total: {len(self)}\n'
                else:
                    lmk += f'Present/Total: {self.len(remove_nan=False)}/{len(self)}\n'
        else:
            for label, coord in self.items():
                if not remove_nan or not _anynan(coord):
                    l = []
                    if keep_label:
                        l += [label]
                    if keep_coordinate:
                        if _anynan(coord):
                            l += [nan_str]*3 
                        else:
                            l += [*map(str,coord)]
                    lmk += [l]

            if separator is None:
                return lmk
            lmk = [separator.join(l).strip(separator) for l in lmk]

            if line_ending is None:
                return lmk
            lmk = line_ending.join(lmk).strip(line_ending)

            if lmk:
                lmk += line_ending
        
        return header + lmk

    def sort_by(self, ordered_labels):
        # select those labels, and order them, filling in nan if necessary
        return self.__class__(
            {label:([*self[label]] if label in self else [_nan]*3) for label in ordered_labels}
        )

    def len(self, remove_nan=False):
        l = len(self)
        if remove_nan:
            l -= sum([ _anynan(v) for v in self.values() ])
        return l

    def __repr__(self):
        return self.string(formatted=True)

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
        import numpy as np
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

        lmk = LandmarkDict(zip(lmk.labels, coords_new))
        lmk.update(lmk_not_moved['Detached'])

        return lmk


class vtkLandmark:

    Color = (1,0,0)
    HighlightColor = (0,1,0)

    def __init__(self, label:str, coord=(0,0,0), **kwargs):
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


def library(db=db_location):
    if '_library' in globals():
        return globals()['_library']
    else:
        try:
            globals()['_library'] = Library()
            return globals()['_library']
        except Exception as e:
            print(f'landmark library is not loaded\n{e}')


# if __name__=='__main__':
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
#                 line_ending=args.read_line_break)
#     lmk_str = lmk.string(
#                 header=args.header,
#                 formatted=args.formatted,
#                 nan_str=args.nan_string,
#                 keep_label=not args.no_label,
#                 keep_coordinate=not args.label_only,
#                 separator=args.delimiter,
#                 line_ending=args.line_break)

#     if args.output is not None:
#         with open(args.output, 'wt', newline='') as f:
#             f.write(lmk_str)
#     else:            
#         sys.stdout.write(lmk_str)

