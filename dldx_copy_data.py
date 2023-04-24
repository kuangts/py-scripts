
import os, sys, glob, csv
import numpy as np
from vtk_bridge import *
from stl import *
from CASS import CASS


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

            f.copy_landmarks(os.path.join(sub_dir, 'lmk_info.txt'))
            lmk = f.load_landmarks(interpreter=lib)
            with open(os.path.join(sub_dir, 'lmk.csv'),'w') as ff:
                csv.writer(ff).writerows([[k] + list(v) for k,v in lmk.items()])


            models_to_load = ['CT Soft Tissue', 'Skull','Mandible','Lower Teeth']

            if f.has_model('Upper Teeth'): # one-piece lefort
                models_to_load.append('Upper Teeth')
            else: # multiple-piece lefort
                models_to_load.append('Upper Teeth (original)')

            ind = f.model_indices(models_to_load)

            if any(map(lambda x:x is None, ind)):

                print(' '.join((f'  {anon_id} is missing models:',*[models_to_load[i] for i,x in enumerate(ind) if x is None])))
                continue

            else:
                
                try:
                    models = f.load_models(models_to_load)
                    
                    for im,m in enumerate(models):

                        # transform stl according to cass rule
                        v4 = np.hstack((m.v, np.ones((m.v.shape[0],1))))
                        v4 = v4 @ np.array(m.T[0]).T
                        m.v[...] = v4[:,:3]

                        # write stl
                        stl_name = models_to_load[im]
                        if stl_name == 'Upper Teeth (original)':
                            stl_name = 'Upper Teeth'
                        stl_name += '.stl'
                        write_stl(m, os.path.join(sub_dir, stl_name))

                        # write two transformations
                        if models_to_load[im] == 'Skull':
                            with open(os.path.join(sub_dir, 'global_t.tfm'), 'w', newline='') as f:
                                csv.writer(f, delimiter=' ').writerows(m.T[0])

                        elif models_to_load[im] == 'Mandible':
                            with open(os.path.join(sub_dir, 'mandible_t.tfm'), 'w', newline='') as f:
                                csv.writer(f, delimiter=' ').writerows(m.T[0])

                except Exception as e:
                    print(e)
                    continue
                
                else:
                    info[i+1] = sub_info
                    try: # in case csv file is locked
                        with open(info_sheet, 'w', newline='') as f:
                            csv.writer(f).writerows(list(info.values()))
                    except: # wait til next iteration to save info sheet
                        continue



if __name__=='__main__':
    from kuang.digitization import library
    lib = library.Library(r'kuang\CASS.db')
    root = os.path.normpath(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\temp\original')
    os.makedirs(os.path.join(os.path.dirname(root), 'export'), exist_ok=True)
    log_file = os.path.join(os.path.dirname(root), 'export', 'log.txt')
    with open(log_file, 'w') as sys.stdout:
        from datetime import datetime
        print(datetime.now())
        print('>>', *sys.argv)
        copy_data(root)
    
