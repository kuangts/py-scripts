import csv
from collections import namedtuple

landmark_entry_fields = ('ID','M','Category','Group','Name','Fullname','Description','Coordinate','View','DP') # corresponding to the db
_LDMK = namedtuple('LDMK', landmark_entry_fields)

class _LDMK(_LDMK):

    def print_str(self):
        d = dict(zip(landmark_entry_fields, (
            f'[{self.ID:>3}]',
            f'{self.M}',
            f'{self.Category}',
            f'{self.Group[3:]:<12}',
            f'{self.Name:<10}',
            f'{self.Fullname.strip()}',
            f'{self.Description.strip()}',
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


def read(file_path=None):
    if file_path is None:
        file_path = r'C:\py-scripts\tools\landmark_definition_research.csv'
    lib = []
    with open(file_path, 'r') as f:             
        for x in csv.reader(f):
            lib.append(_LDMK(*x))
    return lib
