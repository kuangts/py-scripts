import os, sys, pathlib, shutil, tkinter, csv, glob
from tkinter import filedialog, messagebox
from datetime import datetime
import open3d as o3d
import numpy as np
import dicom
import dicom2stl
from dicom2stl import create_stl_model

names = {}
with open(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\soft-tissue-prediction.csv','r') as f:
    for l in csv.reader(f):
        names[int(l[0])] = l[1]

def check_ct_stl(src=r'C:\OneDrive\FEM Mesh Generation\Cases', dst=r'C:\data\xi', cases=None, ct_dir='pre', copy_files = ['mandible_landmark.txt','maxilla_landmark.txt','skin_landmark.txt'], match_stl_files=['mandible_surface.stl','maxilla_surface.stl'], skip_completed_cases=True):

    # convenience method
    completed = lambda subj:(dst/subj/ct_dir).is_dir() and all([(dst/subj/f).is_file() for f in copy_files])
    remove_nesting = lambda input, output: [remove_nesting(i, output) for i in input] if isinstance(input,list) else output.append(input)
    ct2stl = lambda ct: create_stl_model(ct, threshold='bone', new_spacing=None, reset_origin=False, smoothing=False, show_model=False, number_of_triangles=None)

    tkinter.Tk().withdraw()
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    copy_files_flat, match_stl_flat = [],[]
    remove_nesting([copy_files], copy_files_flat)
    remove_nesting([match_stl_files], match_stl_flat)
    match_stl_files = match_stl_flat
    copy_files = copy_files_flat
    if cases is None: 
        cases = range(1,101)

    for n in cases:
        subj = f'n{n:04}'
        subj_name = names[n]
        model_dict = {}
        img_path = ''
        print(f'{subj}:{subj_name}')
        if completed(subj):
            print('  good')
            img_path = dst/subj/ct_dir
            if skip_completed_cases:
                continue

        for stl_name in match_stl_files:
            if stl_name not in model_dict:
                stl = str(src/subj/stl_name)
                if os.path.isfile(stl):
                    model_dict[stl] = o3d.io.read_triangle_mesh(stl).paint_uniform_color((.2,.5,.5)).compute_vertex_normals()
                else:
                    stls = filedialog.askopenfilenames(title=f'SELECT STL {stl_name} for {subj}:{subj_name}')
                    for s in stls:
                        model_dict[s] = o3d.io.read_triangle_mesh(s).paint_uniform_color((.2,.5,.5)).compute_vertex_normals()
        while 1:
            # view existing ct series
            if img_path:
                model_dict.update(
                    ct2stl(img_path)
                )
                o3d.visualization.draw_geometries(list(model_dict.values()), mesh_show_back_face=True)
                if messagebox.askyesno(title=f'{subj}:{subj_name}', message='Proceed to next subject?'):
                    break
            # find ct series
            img_path = filedialog.askdirectory(title=f'SELECT CT for {subj}:{subj_name}')

            if img_path:
                # find subfolder with the most dicom files
                dirs = [os.path.join(img_path,f) for f in os.listdir(img_path) if os.path.isdir(os.path.join(img_path,f))]
                if any(dirs):
                    img_path = dirs[np.array([len(list(os.listdir(d))) for d in dirs]).argmax()]
                temp = '_' + subj + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")
                shutil.copytree(img_path, dst/temp)
                img_path = ''
                model_dict.update(
                    ct2stl(dst/temp)
                )
                o3d.visualization.draw_geometries(list(model_dict.values()), mesh_show_back_face=True)
                # copy this series if right one is found
                if (dst/subj/ct_dir).is_dir():
                    do_copy = messagebox.askyesno(title=f'{subj}:{subj_name}', message='Override existing CT series?')
                    if do_copy: shutil.rmtree(dst/subj/ct_dir)
                else:
                    do_copy = messagebox.askyesno(title=f'{subj}:{subj_name}', message='Copy this CT series?')
                if do_copy:
                    shutil.move(dst/temp, dst/subj/ct_dir)
            # copy files
            for f in copy_files:
                if not (dst/subj/f).is_file():
                    if (src/subj/f).is_file():
                        shutil.copy(src/subj/f, dst/subj)
                    else:
                        sel_file = filedialog.askopenfilenames(title=f'SELECT {f} for {subj}:{subj_name}')
                        if sel_file:
                            for f in sel_file:
                                shutil.copy(f, dst/subj)
                        else:
                            print(f'{f} not found')
                            break
            # stay with this subject or move on to the next
            if messagebox.askyesno(title=f'{subj}:{subj_name}', message='Proceed to next subject?'):
                for f in dst.glob('_'+subj+'*'):
                    shutil.rmtree(f)
                print('  completed' if completed(subj) else '  INCOMPLETE')
                break

def check_ct(src=r'C:\OneDrive\FEM Mesh Generation\Cases', dst=r'C:\data\dicom', cases=None):
    # add button later
    ct2stl = lambda ct: create_stl_model(ct, threshold='bone', new_spacing=None, reset_origin=False, smoothing=False, show_model=False, number_of_triangles=None)
    tkinter.Tk().withdraw()
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    if cases is None: 
        cases = range(1,101)
    for n in cases:
        subj = f'n{n:04}'
        subj_name = names[n]
        print(f'{subj}:{subj_name}')
        while 1:
            # find ct series
            img_path = filedialog.askdirectory(title=f'SELECT CT for {subj}:{subj_name}')
            if img_path:
                # find subfolder with the most dicom files
                dirs = [os.path.join(img_path,f) for f in os.listdir(img_path) if os.path.isdir(os.path.join(img_path,f))]
                if any(dirs):
                    img_path = dirs[np.array([len(list(os.listdir(d))) for d in dirs]).argmax()]
                temp = '_' + subj + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")
                shutil.copytree(img_path, dst/temp)
                info = dicom.read(dst/temp, return_image=False, return_info=True)
                model_dict = ct2stl(dst/temp)
                o3d.visualization.draw_geometries(list(model_dict.values()), mesh_show_back_face=True)

                ans = messagebox.askyesnocancel(title=f'{subj}:{subj_name}', message='Is this pre-op?')
                if ans == True:
                    new_path = str(dst/temp[1:])+'-pre'
                    shutil.move(dst/temp, new_path)
                    with open(r'c:\data\dicom\preop.csv', 'a', newline='') as f:
                        csv.writer(f).writerow([img_path, new_path, subj, info['name'], info['age'], info['Study Date']])
                if ans == False:
                    new_path = str(dst/temp[1:])+'-post'
                    shutil.move(dst/temp, new_path)
                    with open(r'c:\data\dicom\postop.csv', 'a', newline='') as f:
                        csv.writer(f).writerow([img_path, new_path, subj, info['name'], info['age'], info['Study Date']])
                if ans is None:
                    shutil.rmtree(dst/temp)
            # stay with this subject or move on to the next
            else:
                if messagebox.askyesno(title=f'{subj}:{subj_name}', message='Proceed to next subject?'):
                    break



if __name__ == '__main__':
    
    # unc40 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17, 19, 20, 21, 22, 28, 30, 31, 32, 34, 36, 40, 41, 42, 43, 44, 45, 46, 49, 52, 53, 55, 56, 59, 60, 65, 66, 67, 68, 70]
    # all = glob.glob(r'C:\data\dicom\*-post')
    # for i in unc40:
    #     name = f'n{i:04}'
    #     print(name)
    #     file = glob.glob(rf'C:\data\dicom\{name}-*-post')
    #     for i in file:
    #         info, filenames = dicom.read_info(i, return_filenames=True)
    #         all_dcm = glob.glob(i+'\\*')
    #         del_dcm = [x for x in all_dcm if x not in filenames]
    #         for x in del_dcm:
    #             os.remove(x)

    #         stl_name = os.path.join(r'C:\data\stls', name+'-'+info['Study Date']+'-bone.stl')
    #         stl = dicom2stl.create_stl_model(i, threshold='bone', show_model=False,number_of_triangles=100_000)
    #         o3d.io.write_triangle_mesh(stl_name, stl['bone'])
    #     if len(file)==1:
    #         x,y = os.path.splitext(stl_name)
    #         shutil.move(stl_name, x+'-post'+y)


    all = glob.glob(r'C:\data\pre-post-paired\*\*-pre')
    for i in all[5:]:
        name, stl_name = os.path.split(i)
        name = os.path.basename(name)[:5]
        print(name)

        _, filenames = dicom.read_info(i, return_filenames=True)

        all_dcm = glob.glob(i+'\\*')
        del_dcm = [x for x in all_dcm if x not in filenames]
        if len(del_dcm)>1 and input(f'remove {len(del_dcm)} out of {len(all_dcm)} files: [y]es or [n]o\n').lower()=='y':
            for x in del_dcm:
                os.remove(x)

        stl_name = os.path.join(r'C:\data\stls', name+'-'+stl_name+'-bone.stl')
        stl = dicom2stl.create_stl_model(i, threshold='bone', show_model=False, number_of_triangles=100_000)
        o3d.io.write_triangle_mesh(stl_name, stl['bone'])

    # # post-op
    # for n in range(1,71):
    #     subj = f'n{n:04}'
    #     print(subj)
    #     if not (dst/subj).is_dir():
    #         continue        
    #     # # match with postop segmented stl
    #     # stl_files = filedialog.askopenfilenames()
    #     # stl = []
    #     # for f in stl_files:
    #     #     stl.append(o3d.io.read_triangle_mesh(str(f)).paint_uniform_color((.2,.5,.5))).compute_vertex_normals()
    #     next = False
    #     while not next:
    #         img_path = filedialog.askdirectory()
    #         temp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #         shutil.copytree(img_path, dst/temp)
    #         m = create_stl_model(dst/temp, threshold='bone', new_spacing=(1.,1.,1.), reset_origin=False, smoothing=False, show_model=False, number_of_triangles=10000)
    #         # reg_tfm = filedialog.askopenfilename()
    #         # if reg_tfm:
    #         #     with open(reg_tfm,'r') as f:
    #         #         reg = np.loadtxt(f.read().split('Units')[0].split('\n'))
    #         #     for x in m:
    #         #         x.transform(reg)
    #         o3d.visualization.draw_geometries(list(m.values()), mesh_show_back_face=True)
    #         response = input('[c]opy this series, [p]roceed to next subject, [s]elect other series of the same subject\n')
    #         if response == 'c':
    #             shutil.move(dst/temp, dst/subj/'post')
    #             print('copied')
    #         elif response == 'p':
    #             shutil.rmtree(dst/temp)                
    #             next = True
    #         else:
    #             shutil.rmtree(dst/temp)

