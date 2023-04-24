
import os, sys, glob, csv
import numpy as np
from vtk_bridge import *
from cass_bridge import *
from stl import *
from kuang.digitization import library
lib = library.Library(r'kuang\CASS.db')


def copy_data(root):

    export_dir = os.path.join(os.path.dirname(os.path.normpath(root)), 'export')
    info_sheet = os.path.join(export_dir, 'info.csv')
    cass_files = glob.glob(os.path.join(root, '*.cass'))
    cass_files.sort()
    info = {}
    cass_files_exclude = []


    if os.path.exists(info_sheet):
        with open(info_sheet, 'r') as f:
            info = {i+1:x for i,x in enumerate(csv.reader(f))}
            cass_files_exclude = [x[-1] for x in info.values()]
        print(f'total {len(cass_files_exclude)} finished: ')
        for c in cass_files_exclude:
            print(c)

    for i,sub in enumerate(cass_files):

        if os.path.basename(sub) in cass_files_exclude:
            continue

        with CASS(sub) as f:

            anon_id = f'DLDX{i+1:03}'
            sub_info = [anon_id, *f.subject_info, os.path.basename(sub)]
            print(f'{i+1}:{sub_info}')                    
            sub_dir = os.path.join(export_dir, anon_id)
            os.makedirs(sub_dir, exist_ok=True)

            f.write_landmarks(os.path.join(sub_dir, 'lmk.csv'))

            model_list = ['CT Soft Tissue', 'Skull','Mandible','Lower Teeth']

            if f.has_model('Upper Teeth'): # one-piece lefort
                model_list.append('Upper Teeth')
            else: # multiple-piece lefort
                model_list.append('Upper Teeth (original)')

            f.copy_models(model_list, sub_dir, transform=True, allow_override=True, user_select=False)
            if os.path.exists(os.path.join(sub_dir, 'Upper Teeth (original).stl')):
                shutil.move(os.path.join(sub_dir, 'Upper Teeth (original).stl'), os.path.join(sub_dir, 'Upper Teeth.stl'))
            # write two transformations
            f.write_transformation('Skull', os.path.join(sub_dir, 'global_t.tfm'))
            f.write_transformation('Mandible', os.path.join(sub_dir, 'mandible_t.tfm'))

            info[i+1] = sub_info
            try: # in case csv file is locked
                with open(info_sheet, 'w', newline='') as f:
                    csv.writer(f).writerows(list(info.values()))
            except: # wait til next iteration to save info sheet
                continue



if __name__=='__main__':
    root = os.path.normpath(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\temp\original')
    os.makedirs(os.path.join(os.path.dirname(root), 'export'), exist_ok=True)
    cass_files = glob.glob(os.path.join(root, '*.cass'))
    cass_files.sort()
    export_dir = os.path.join(os.path.dirname(os.path.normpath(root)), 'export')
    log_file = os.path.join(os.path.dirname(root), 'export', 'log.txt')

    # with open(log_file, 'w') as sys.stdout:
    #     from datetime import datetime
    #     print(datetime.now())
    #     print('>>', *sys.argv)
    
    for i,sub in enumerate(cass_files):

        with CASS(sub) as f:

            anon_id = f'DLDX{i+1:03}'
            sub_info = [anon_id, *f.subject_info, os.path.basename(sub)]
            print(f'{i+1}:{sub_info}')                    
            sub_dir = os.path.join(export_dir, anon_id)
            if os.path.exists(sub_dir):
                continue

            os.makedirs(sub_dir, exist_ok=True)

            # write two transformations
            f.write_transformation('Skull', os.path.join(sub_dir, 'global_t.tfm'))
            f.write_transformation('Mandible', os.path.join(sub_dir, 'mandible_t.tfm'))


            f.db = lib
            f.write_landmarks(os.path.join(sub_dir, 'lmk.csv'))

            model_list = ['CT Soft Tissue', 'Skull','Mandible','Lower Teeth']

            if f.has_model('Upper Teeth'): # one-piece lefort
                model_list.append('Upper Teeth')
            else: # multiple-piece lefort
                model_list.append('Upper Teeth (original)')

            f.copy_models(model_list, sub_dir, transform=True, fail_if_exists=False, user_select=False)

            if not all(map(lambda f:os.path.exists(os.path.join(sub_dir, f+'.stl')), model_list)):
                shutil.rmtree(sub_dir)
                print(f'{sub} copy failed')
                continue

            if 'Upper Teeth (original).stl' in model_list:
                shutil.move(os.path.join(sub_dir, 'Upper Teeth (original).stl'), os.path.join(sub_dir, 'Upper Teeth.stl'))


