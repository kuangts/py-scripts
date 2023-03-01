import os, sys

os.chdir(os.path.dirname(__file__))
sys.path = [r'c:\py-scripts'] + sys.path

from kuang.digitization import *

lib = default_library.set_coordinates('A', (2,3,4), default_coordinates=None)
print(lib.find(Name='A'))

lmk_csv = landmark.from_text('n09_.csv')
print(lmk_csv)
print(len(lmk_csv))

# lmk_csv.set('A',(1,3,5))
# print(lmk_csv.get('A'))
# print(lmk_csv['coordinates'])

print(lib.find(Name='A'))
lib = lib.set_coordinates(lmk_csv['name'], lmk_csv['coordinates'], default_coordinates=None)
print(lib.find(Name='A'))

# lmk_xls = landmark.from_excel('n09.xlsx')
# print(lmk_xls)
# print(len(lmk_xls))
# lmk_cass = landmark.from_cass('n09.cass', interpreter=default_library)
# print(lmk_cass)
# print(len(lmk_cass))


