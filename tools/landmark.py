import os
import re, csv
import numpy as np
from pandas import read_excel
from .ui import MODE, PolygonalSurfacePointSelector
from .polydata import polydata_from_stl

class Landmark(dict): 

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
        return { n:v for n,v in self.items() if "--post" in n}
            

def landmark_from_file(file):
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

    return _reg_match(lines)


def landmark_from_excel(file):
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
    return Landmark(zip(labels, zip(*coords)))

        

def write_landmark_to_file(lmk, file):
    with open(file, 'w', newline='') as f:
            
        writer = csv.writer(f)
        for k,v in lmk.items():
            writer.writerow([k,*v])
        return None


class Digitizer(PolygonalSurfacePointSelector):


    def initialize(self, stl_path_or_polydata, mode=None, read_path='', name_list='', save_path='', override=False):
        self.read_path = read_path
        self.override = override
        self.save_path = save_path if save_path else read_path

        # check save_path
        if os.path.exists(self.save_path) and not os.path.isfile(self.save_path):
            self.save_path = None
            print('ignoring save_path')

        # if read_path or list of landmark labels is given
        # run it in editing mode
        if read_path or name_list:

            # initialize nan landmarks if only a list of landmark names are given
            if name_list and not read_path:
                lmk = Landmark(zip(name_list, ((float('nan'),)*3,)*len(name_list)))

            # if a file is given, read the file
            else:
                lmk = landmark_from_file(read_path)
            
            super().initialize(mode if mode else MODE.EDIT, named_points=lmk)
        
        # otherwise, run it in adding mode
        else:

            super().initialize(mode if mode else MODE.FREE, named_points=Landmark())

        if isinstance(stl_path_or_polydata, str):
            self.add_pick_polydata(polydata_from_stl(stl_path_or_polydata))
        else:
            self.add_pick_polydata(stl_path_or_polydata)

        return None


    def save(self):
        if self.save_path:
            if os.path.exists(self.save_path) and not os.path.isfile(self.save_path):
                self.save_path = None
                print('ignoring save_path')

            elif self.override or not os.path.exists(self.save_path):
                with open(self.save_path, 'w', newline='') as f:
                    self.save_file(f)
                return None
        
        self.save_ui(
            title='Save landmarks to *.csv ...',
            initialdir=os.path.dirname(self.read_path if self.read_path else None),
            initialfile=os.path.basename(self.read_path if self.read_path else None)
        )
        
        return None


    def save_file(self, f):
        writer = csv.writer(f)
        for i in range(self.selection_points.GetNumberOfPoints()):
            writer.writerow([self.selection_names[i], *self.selection_points.GetPoint(i)])
        return None


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
    return Landmark(zip(labels, coords))


def _test():
        

    test_string_1 = '''  

        S -4.5347518e+01  -5.8202733e+01   1.0791650e+02


        A1-R,-1.5234984e+01  -6.8431947e+01   1.0830880e+02
        AA ,2.1189399e+01  -6.7932851e+01   1.0831665e+02;


        4.9391600e+01,  -5.5878316e+01 ,  1.0970158e+02
        -2.0258200e+01  ,-5.8505100e+01   3.9840100e+01
    \t2.6734600e+01  -5.9040300e+01   3.9676500e+01
        -1.4645683e+01  -6.5116002e+01   7.1792917e+01
        2.0794633e+01  -6.3329100e+01   7.2100732e+01
        3.7807000e+00  -8.5230700e+01   7.2821200e+01
        -1.2006600e+01  -6.9701764e+01   7.7448631e+01
        1.9210966e+01  -6.6078583e+01   7.7822100e+01
        -6.1532000e+00  -7.6784800e+01   6.8160400e+01
        1.2700000e+01  -7.6540700e+01   6.7297500e+01
        3.9470000e+00  -7.1555300e+01   6.3274700e+01
        3.6124000e+00  -7.2147600e+01   5.6500400e+01
        3.8993000e+00  -7.4307200e+01   4.4822300e+01
        -2.5329000e+00  -7.4261400e+01   5.1332000e+01
        9.7173999e+00  -7.4588282e+01   5.1269416e+01
        4.6309166e+00  -7.1682966e+01   3.0129817e+01
        -9.6763000e+00  -6.7916400e+01   3.2282400e+01
        1.9405750e+01  -6.7337983e+01   3.1545500e+01
        4.1121001e+00  -6.5919800e+01   2.3530999e+01
        4.4777000e+00  -6.1147900e+01   9.0998000e+00
        4.7119000e+00  -4.2641700e+01   1.1468000e+00
        -2.0015100e+01  -4.5998000e+01   1.0747500e+01
        2.7738600e+01  -4.6850600e+01   1.1028700e+01
        -4.5238334e+00  -7.3027734e+01   1.0736685e+02
        8.1836165e+00  -7.2799269e+01   1.0746563e+02
        1.7311000e+00  -7.6514000e+01   1.1729630e+02
        1.6545000e+00,-8.1651100e+01,1.2953400e+02;
        -4.8691800e+01  -5.8412401e+01   9.0010300e+01
        5.1627400e+01  -5.8410500e+01   9.0347500e+01
        -7.6116900e+01   1.6348400e+01   1.1960410e+02
        -5.8733749e+01   1.7223450e+01   7.2437649e+01
        7.3702000e+01   1.6503100e+01   1.1907440e+02
        5.9607166e+01   2.0530834e+01   7.3532931e+01
        -2.7664900e+01  -6.5254800e+01   1.0171310e+02
        3.5552034e+01  -6.3578166e+01   1.0349850e+02
        -2.7934534e+01  -7.3063281e+01   1.1423792e+02
        3.6528467e+01  -7.0686968e+01   1.1558643e+02
        -4.4819300e+01  -1.7795700e+01   2.8266800e+01
        5.0144100e+01  -1.7920100e+01   3.0196900e+01
        3.8611000e+00  -8.3272700e+01   8.1965700e+01
        -3.0746866e+01  -7.5977819e+01   1.2705350e+02
        3.2436600e+01  -7.5895833e+01   1.2965323e+02
        4.3287000e+00  -6.9897300e+01   4.0226900e+01
        4.1276333e+00  -7.4967036e+01   5.1366667e+01
        -6.8808366e+01   1.1463817e+01   9.3021933e+01
        6.6803900e+01   1.4157583e+01   9.1843899e+01
        -3.6280833e+00  -6.9089952e+01   4.1138199e+01
        9.9853000e+00  -6.9218200e+01   4.1054700e+01
        -1.1374366e+01  -6.3966867e+01   4.0902934e+01
        2.0248317e+01  -6.4135118e+01   4.1027534e+01
    '''

    test_string_2 = '''  -4.5347518e+01  -5.8202733e+01   1.0791650e+02
  -1.5234984e+01  -6.8431947e+01   1.0830880e+02
   2.1189399e+01  -6.7932851e+01   1.0831665e+02
   4.9391600e+01  -5.5878316e+01   1.0970158e+02
  -2.0258200e+01  -5.8505100e+01   3.9840100e+01
   2.6734600e+01  -5.9040300e+01   3.9676500e+01
  -1.4645683e+01  -6.5116002e+01   7.1792917e+01
   2.0794633e+01  -6.3329100e+01   7.2100732e+01
   3.7807000e+00  -8.5230700e+01   7.2821200e+01
  -1.2006600e+01  -6.9701764e+01   7.7448631e+01
   1.9210966e+01  -6.6078583e+01   7.7822100e+01
  -6.1532000e+00  -7.6784800e+01   6.8160400e+01
   1.2700000e+01  -7.6540700e+01   6.7297500e+01
   3.9470000e+00  -7.1555300e+01   6.3274700e+01
   3.6124000e+00  -7.2147600e+01   5.6500400e+01
   3.8993000e+00  -7.4307200e+01   4.4822300e+01
  -2.5329000e+00  -7.4261400e+01   5.1332000e+01
   9.7173999e+00  -7.4588282e+01   5.1269416e+01
   4.6309166e+00  -7.1682966e+01   3.0129817e+01
  -9.6763000e+00  -6.7916400e+01   3.2282400e+01
   1.9405750e+01  -6.7337983e+01   3.1545500e+01
   4.1121001e+00  -6.5919800e+01   2.3530999e+01
   4.4777000e+00  -6.1147900e+01   9.0998000e+00
   4.7119000e+00  -4.2641700e+01   1.1468000e+00
  -2.0015100e+01  -4.5998000e+01   1.0747500e+01
   2.7738600e+01  -4.6850600e+01   1.1028700e+01
  -4.5238334e+00  -7.3027734e+01   1.0736685e+02
   8.1836165e+00  -7.2799269e+01   1.0746563e+02
   1.7311000e+00  -7.6514000e+01   1.1729630e+02
   1.6545000e+00  -8.1651100e+01   1.2953400e+02
  -4.8691800e+01  -5.8412401e+01   9.0010300e+01
   5.1627400e+01  -5.8410500e+01   9.0347500e+01
  -7.6116900e+01   1.6348400e+01   1.1960410e+02
  -5.8733749e+01   1.7223450e+01   7.2437649e+01
   7.3702000e+01   1.6503100e+01   1.1907440e+02
   5.9607166e+01   2.0530834e+01   7.3532931e+01
  -2.7664900e+01  -6.5254800e+01   1.0171310e+02
   3.5552034e+01  -6.3578166e+01   1.0349850e+02
  -2.7934534e+01  -7.3063281e+01   1.1423792e+02
   3.6528467e+01  -7.0686968e+01   1.1558643e+02
  -4.4819300e+01  -1.7795700e+01   2.8266800e+01
   5.0144100e+01  -1.7920100e+01   3.0196900e+01
   3.8611000e+00  -8.3272700e+01   8.1965700e+01
  -3.0746866e+01  -7.5977819e+01   1.2705350e+02
   3.2436600e+01  -7.5895833e+01   1.2965323e+02
   4.3287000e+00  -6.9897300e+01   4.0226900e+01
   4.1276333e+00  -7.4967036e+01   5.1366667e+01
  -6.8808366e+01   1.1463817e+01   9.3021933e+01
   6.6803900e+01   1.4157583e+01   9.1843899e+01
  -3.6280833e+00  -6.9089952e+01   4.1138199e+01
   9.9853000e+00  -6.9218200e+01   4.1054700e+01
  -1.1374366e+01  -6.3966867e+01   4.0902934e+01
   2.0248317e+01  -6.4135118e+01   4.1027534e+01'''
    x = _reg_match(test_string_2)
    return x
    # x = landmark_from_file(r'P:\Kevin Gu - DO NOT TOUCH\soft tissue prediction (pre-post-paired-40-send-copy-0321)\pre-post-paired-40-send-copy-0321\n0001\skin_landmark.txt')
    # print(x)
    # x = landmark_from_file(r'C:\data\pre-post-paired-soft-tissue-lmk-23\n0001\skin-pre-23-cass.txt')
    print(x)
    print(x.midline())


def _test_excel():
    landmark_from_excel(r'.\test\lmk.xlsx')


if __name__ == '__main__':
    _test_excel()