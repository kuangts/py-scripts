
import os, sys, glob, csv
import numpy as np
from vtk_bridge import *
from stl import *
from CASS import CASS


def copy_lmk(root):

    export_dir = os.path.join(os.path.dirname(os.path.normpath(root)), 'export')
    cass_files = glob.glob(os.path.join(root, '*.cass'))
    cass_files.sort()

    for i,sub in enumerate(cass_files):

        with CASS(sub) as f:

            anon_id = f'DLDX{i+1:03}'
            sub_dir = os.path.join(export_dir, anon_id)
            os.makedirs(sub_dir, exist_ok=True)
            f.copy_landmarks(os.path.join(sub_dir, 'lmk_info.txt'))
            print(anon_id)
            


if __name__=='__main__':

    root = os.path.normpath(sys.argv[1])
    copy_lmk(root)
    
