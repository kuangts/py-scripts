import re
from collections.abc import Sequence

import numpy as np
import pandas
import rarfile

from ..geo3d.point import NamedArray


class Landmark(NamedArray):
    '''this class is one of the two primary ways of storing landmark information, the other being `Library`
    it facilitates calculation using landmark coordinates without changing size or order of the ladnarmk set
    it provides get and set methods for working with coordinates of existing landmarks
    but does not allow automatic insert similar to python dictionary
    use this class when calculation is the main concern; use `Library` to access definition for each landmark
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
        V = pandas.read_excel(file, header=0, engine='openpyxl').values
        return cls(zip(V[1:,0], V[1:,1:4]+V[0,1:4]))

    @classmethod
    def from_cass(cls, file, interpreter=None):
        with rarfile.RarFile(file) as f:
            with f.open('Measure_Data_Info_new.bin') as l:
                t = l.read().decode("utf-8").strip(';').split(';')
                lmk = [x.split(',') for x in t]
                lmk = {int(l[0]):tuple(map(float,l[-3:])) for l in lmk}
        
        if interpreter is not None:
            lmk_new = {}
            lmk_del = {}
            for k,v in lmk.items():
                x = interpreter.find(ID=k)
                if x is not None:
                    lmk_new[x.Name] = v
                else:
                    lmk_del[k] = v
            lmk = lmk_new

        return cls(lmk)

    def get(self, name):
        try:
            id = (self['name']==name).nonzero()[0]
            if not len(id):
                raise ValueError(f'cannot find {name}')
            coord = self[(self['name']==name).nonzero()[0][0]]['coordinates']
        except Exception as e:
            print(e)
            coord = np.asarray([np.nan,]*3)
        return coord

    def set(self, name, coord):
        try:
            id = (self['name']==name).nonzero()[0]
            if not len(id):
                raise ValueError(f'cannot find {name}')
        except:
            return np.asarray([np.nan,]*3)

        old_coord = self[id[0]]['coordinates'].copy() ## WHY?????????????????
        self[id[0]]['coordinates'][:] = coord[:]
        return old_coord

    def select(self, ordered_labels):
        arr = np.empty((len(ordered_labels),3))
        arr[:] = np.nan
        _, ind1, ind2 = np.intersect1d(self['name'], ordered_labels)
        arr[ind2,:] = self.coordinates[ind1,:]
        return self.__class__(zip(ordered_labels, arr))


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
                ordered_label = self['name'] 
            for label, coord in ordered_label:
                x = []
                if keep_label:
                    x += [label]
                if keep_coordinate:
                    x += [*map(str,coord)]
                lmk += [x]

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

        lmk = self.__class__(zip(lmk.keys(), coords_new))
        lmk.update(lmk_not_moved['Detached'])

        return lmk






# class LandmarkSet(Library):
#     """
#     immutable set containing immutable landmark entries including coordinates
#     this class uses the same structure as `Library`
#     but reads landmark coordinates from file.
#     IMMUTABLE and UNORDERED
#     """

#     def __new__(cls, file_location_or_existing_set, db_location=default_db_location, **parseargs):
#         # read the db to get landmark definitions
#         db = super().__new__(cls, db_location)
#         if isinstance(file_location_or_existing_set, str):
#             if not file_location_or_existing_set:
#                 coord_dict = {}
#             else:
#                 # read landmark string
#                 with open(file_location_or_existing_set, 'r') as f:
#                     lmk_str = f.read()
#                 # parse landmark string
#                 labels, coordinates, _ = LandmarkDict.parse(lmk_str, **parseargs)
#                 coord_dict = dict(zip(labels, coordinates))
#                 # check if there is unknown landmark
#                 if not all(x in db.field('Name') for x in coord_dict):
#                     print('WARNING: some labels are not recognized')
#             # prepare for instantiation
#             lmk_set = {x._replace(Coordinate=tuple(coord_dict[x.Name])) for x in db}
#         else:
#             lmk_set = file_location_or_existing_set

#         # create landmark set
#         obj = super().__new__(cls, lmk_set)

#         return obj
            

#     def __getitem__(self, __key):
#         if isinstance(__key, str):
#             x = self.find(Name=__key)
#             if x is None:
#                 return (float('nan'),)*3
#             return x.Coordinate
#         elif isinstance(__key, Sequence) and all(isinstance(x,str) for x in __key):
#             return [self.__getitem__(k) for k in __key]
#         else:
#             raise ValueError('cannot get item: wrong key')
            
#     def set_coordinates(self, __key, __value):
#         if isinstance(__key , str):
#             return self.set_coordinates([__key],[__value])
#         elif isinstance(__key, Sequence) and isinstance(__value, Sequence) and len(__key) == len(__value) and all(isinstance(x, str) for x in __key) and all(len(v)==3 for v in __value):
#             return self.__class__({x._replace(Coordinate=tuple(__value[__key.index(x.Name)])) if x.Name in __key else x for x in self})
#         else:
#             raise ValueError('cannot set coordinates')

#     def __setitem__(self, __key, __value) -> None:
#         raise ValueError('setting item not allowed for immutable set')

#     def __delitem__(self, __key) -> None:
#         raise ValueError('deleting item not allowed for immutable set')

#     @property
#     def centroid(self):
#         return np.asarray(list(self.field('Coordinate'))).mean(axis=0).tolist()

#     def translate(self, t):
#         labels, coords = self.split(('Name','Coordinate'))
#         coords_dict = dict(zip(labels, (np.asarray(coords) + t).tolist()))
#         return self.__class__({ x._replace(Coordinate=tuple(coords_dict[x.Name])) for x in self })

#     def transform(self, T, post_multiply=True): 
#         T = np.array(T)
#         if T.shape not in ((3,3),(4,4)):
#             raise ValueError(f'incompatible matrix shape: {T.shape} ')
#         labels, coords = self.split(('Name','Coordinate'))
#         if T.shape == (4,4):
#             coords = np.hstack((coords, np.ones((len(coords),1))))
#         new_coords = coords @ T if post_multiply else coords @ T.T
#         coords_dict = dict(zip(labels, new_coords[:,:3].tolist()))
#         return self.__class__({ x._replace(Coordinate=tuple(coords_dict[x.Name])) for x in self })

#     def __repr__(self):
#         return self.string(sort_key=attrgetter('ID'), group_key=None, group=False, id=False, definition=False)


if __name__=='__main__':

    lmk = Landmark.from_text(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\extracted_from_cass.csv')
    # lmk = lmk.bilateral()
    # lmk.transform(np.eye(3))
    # lmk.transform(np.eye(4)).field('Coordinate')
    # lmka = lmk.set_coordinates('A',(1,2,3))
    # print(lmk['A'])
    # print(lmka['A'])
    # print(lib)
    print(len(lmk))
    print(lmk)
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

    t = np.random.rand(3,)
    R = np.random.rand(3,3)
    c = np.random.rand(3,).tolist()
    PP = Landmark([*zip(iter('abcdefghij'),np.random.rand(10,3))])
    premul = True
    def test(x):
        x.translate(t)
        x.rotate(R, pre_multiply=premul, center=c)

    test(PP)
    print(PP)

    print(len(PP[2]['name']), type(PP[2]['name']), isinstance(PP[2]['name'], str), isinstance(PP[2]['coordinates'][2], float))

    PPnew = PP.append(('x',(1,2,3)))
    print(PPnew)
    print(PPnew.dtype)
    k = NamedArray().append(('x',(1,2,3)),('y',(4,5,6)))
    print(k.dtype)
    print(PP.get('a'))
    print(PP.set('z',(4,5,6)))
    print(PP)