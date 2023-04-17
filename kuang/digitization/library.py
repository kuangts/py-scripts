import os, json, sqlite3
from typing import Any
from collections import namedtuple
from collections.abc import Sequence, Iterable
from operator import attrgetter

import numpy

class Library(frozenset):
    """
    immutable set containing immutable landmark definition
    this class is to bridge landmark database (CASS.db) and landmark manipulation in python.
    it provides simple functions for look-up and filter operations.
    this is a subclass of Set/forzenset and therefore it is, again
    IMMUTABLE and UNORDERED
    """

    landmark_entry_fields = ('ID','M','Category','Group','Name','Fullname','Description','Coordinate','View','DP') # corresponding to the db

    LDMK = namedtuple('LDMK', landmark_entry_fields)


    def set_coordinates(self, name, coordinates, default_coordinates=(float('nan'),)*3):
        if isinstance(name , str):
            return self.set_coordinates([name],[coordinates], default_coordinates=default_coordinates)
        else:
            if (isinstance(name, Sequence) or isinstance(name, numpy.ndarray)) and (isinstance(coordinates, Sequence) or isinstance(coordinates, numpy.ndarray))\
                    and len(name) == len(coordinates) and \
                    all(isinstance(x, str) for x in name) and all(len(v)==3 for v in coordinates):
                coordinates = numpy.array(coordinates)
                new_set = set()
                for x in self:
                    if x.Name == 'A':
                        print('here')
                        pass
                    # print(x.Name, x.Coordinate)
                    try:
                        v = tuple(coordinates[name.index(x.Name),:])
                    except:
                        v = x.Coordinate if default_coordinates is None else default_coordinates
                    print(x.Name, v)
                
                    new_set.add(x._replace(Coordinate=v))
            #     lmk_set.__repr__ = lambda x:x.string(sort_key=attrgetter('ID'), group_key=None, group=False, id=False, definition=False)
                return self.__class__(new_set)
            else:
                raise ValueError('cannot set coordinates')


    def __new__(cls, db_location_or_existing_set=''):
        if isinstance(db_location_or_existing_set, cls):
            return db_location_or_existing_set
        elif isinstance(db_location_or_existing_set, str): 
            location = db_location_or_existing_set
            if not location:
                try:
                    location = cls.db_location
                except Exception as e:
                    raise ValueError('cannot find db')
            with sqlite3.connect(location) as con:
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
            if 'New GT' in category:
                del category['New GT'] # because m_order has duplicates

            # after assuring each landmark (lmk) belongs to one and only one group (grp)
            lmk_group = {lmk:grp for grp, mem in category.items() for lmk in mem.values() if lmk }

            # following items are the content of the library
            items = [
                cls.LDMK(
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

            # create instance and record location
            obj = super().__new__(cls, items)
            if not hasattr(cls, 'lib'):
                cls.lib = {}
            if not location in cls.lib:
                cls.lib[location] = obj
                print('library loaded')


        else: # creating library from existing set, likely a subset of the full library
            items = db_location_or_existing_set
            if not all(isinstance(x, cls.LDMK) for x in items):
                raise ValueError(f'cannot create library with the input set')
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
            if fieldname not in self.LDMK._fields:
                raise ValueError(f'{fieldname} is not a valid field')
            return { getattr(x, fieldname) for x in self }
        else:
            raise ValueError('fieldname is not valid input')
    
    def split(self, fieldnames):
        # extracts fields into lists
        # unordered, but correspondence is kept
        # usage: name, coord = self.split(['Name','Coordinate'])
        if isinstance(fieldnames, str):
            raise ValueError(f'use field() instead to extract single field')
        else:
            for f in fieldnames:
                if f not in self.LDMK._fields:
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
    
    
    class LDMK(LDMK):

        def print_str(self):
            d = dict(zip(Library.landmark_entry_fields, (
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


if __name__=='__main__':
    lib = Library(r'kuang/cass.db')
    