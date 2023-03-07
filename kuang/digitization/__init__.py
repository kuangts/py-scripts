import os.path as __path
from json import load as __load
from .library import Library
from .landmark import Landmark


with open(__path.join(__path.dirname(__file__), 'config.json'), 'r') as f:
    _defaults = __load(f)

class Library(Library):
    db_location = _defaults['default_db_location']
    def soft_tissue_23(self): 
        return self.filter(lambda x: x.Name in _defaults['soft_tissue_labels_23'])

default_library = Library()

__ALL__ = ["Landmark", "Library", "default_library"]

