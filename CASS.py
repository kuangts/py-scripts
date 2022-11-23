import rarfile, shutil, os, glob
import numpy as np
import open3d as o3d
import trimesh
from tkinter import messagebox
import tkinter
from datetime import datetime as dt
tkinter.Tk().withdraw()


class CASS(rarfile.RarFile):

    def model_names(file, print_names=True):
        with file.open('Three_Data_Info.bin') as f:
            t = f.read().decode("utf-8").strip(';').split(';')
            names = [x.split(',')[0] for x in t[1:]]
            assert t[0] == str(len(names))
            if print_names:
                for i,n in enumerate(names):
                    print(f'{i:>5} - {n}')
            return names


    def models(file, names=[], dest_dir=None, override=True):
        all_model_names = file.model_names(print_names=False)
        temp_dir = rf'c:\_temp\{dt.now().__str__()}'.replace(' ','-').replace(':','').split('.',1)[0]
        os.makedirs(temp_dir, exist_ok=True)
        if dest_dir is None:
            result = []
        else:
            os.makedirs(dest_dir, exist_ok=True)
            result = None

        for m in names:
            if dest_dir is not None:
                dest = os.path.join(dest_dir, m+'.stl')
                if not override and os.path.isfile(dest):
                    print(f'{dest} exists, skipped')
                    continue
            if m in all_model_names:
                id = all_model_names.index(m)
            else:
                file.model_names(print_names=True)
                id = input(f'type in index to replace \'{m}\':\n')
                try:
                    id = int(id)
                    m = all_model_names[id]                    
                except Exception as e:
                    print(e)
                    result.append(None)
                    print(f'skipping {m}')

            stl_name = str(id) + '.stl'
            file.extract(stl_name, temp_dir)
            if dest_dir is None:
                result.append(o3d.io.read_triangle_mesh(os.path.join(temp_dir, stl_name)))
            else:
                shutil.move(os.path.join(temp_dir, stl_name), dest)
                print(f'{m} found and copied')
        shutil.rmtree(temp_dir)
        return result


    def landmarks(file, index=[]):
        with file.open('Measure_Data_Info_new.bin') as f:
            t = f.read().decode("utf-8").strip(';').split(';')
            lmk = {}
            for xx in [x.split(',') for x in t]:
                id = int(xx[0])
                if id in index:
                    lmk[id] = np.array(xx[-3:], dtype=float)
            return lmk
            


def job_20221118(CASS_file, dst_dir):
    subj = os.path.basename(CASS_file)[:-5]
    print(subj)
    dst_dir = os.path.join(dst_dir, subj)
    f = open_cass(CASS_file)
    extract_stl(f, ['Skull (whole)', 'Mandible (whole)', 'CT Soft Tissue'], dst_dir)
    lmk = extract_lmk(f, {10:'Gb',112:'Go-R',113:'Go-L',158:'C'})
    if any([x is None for x in lmk.values()]):
        print('abort, incomplete landmark')
        return
    skin = o3d.io.read_triangle_mesh(os.path.join(dst_dir, 'CT Soft Tissue.stl'))
    Go_mid = (lmk[112]+lmk[113])/2
    v = np.cross((lmk[158]-Go_mid),[1.,0.,0.])
    v = v/(v**2).sum()**.5
    vtx, fcs = np.asarray(skin.vertices), np.asarray(skin.triangles)
    vtx, fcs = trimesh.intersections.slice_faces_plane(vertices=vtx, faces=fcs, plane_normal=np.asarray([0,0,-1]), plane_origin=lmk[10])
    vtx, fcs = trimesh.intersections.slice_faces_plane(vertices=vtx, faces=fcs, plane_normal=np.asarray([0,-1,0]), plane_origin=Go_mid)
    vtx, fcs = trimesh.intersections.slice_faces_plane(vertices=vtx, faces=fcs, plane_normal=v, plane_origin=Go_mid)
    skin = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vtx),
        triangles=o3d.utility.Vector3iVector(fcs),
    ).compute_vertex_normals()

    o3d.visualization.draw_geometries([skin])

    if messagebox.askyesno(title='', message='Save (override) this surface?'):
        o3d.io.write_triangle_mesh(os.path.join(dst_dir, 'CT Soft Tissue.stl'), skin)

if __name__ == '__main__':
    # for f in glob.glob(r'C:\data\normal_ct_anon\*.CASS'):
    #     job_20221118(f, r'C:\data\NCT')
    job_20221118(r'C:\data\normal_ct_anon\SH001.CASS', r'C:\data\NCT')