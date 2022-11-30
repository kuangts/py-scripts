import rarfile, shutil, os, json, pandas
import open3d as o3d
from datetime import datetime as dt


class CASS(rarfile.RarFile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setattr(self, 'temp_dir', rf'c:\_temp_cass\{dt.now().__str__()}'.replace(' ','-').replace(':','').split('.',1)[0])
        os.makedirs(self.temp_dir, exist_ok=False)


    def close(self, *args, **kwargs):
        super().close()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    def model_names(self, print_names=True):
        with self.open('Three_Data_Info.bin') as f:
            t = f.read().decode("utf-8").strip(';').split(';')
            names = [x.split(',')[0] for x in t[1:]]
            assert t[0] == str(len(names))
            if print_names:
                for i,n in enumerate(names):
                    print(f'{i:>5} - {n}')
            return names


    def models(self, names=[], dest_dir=None, override=True):
        all_model_names = self.model_names(print_names=False)
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
                self.model_names(print_names=True)
                id = input(f'type in index to replace \'{m}\':\n')
                try:
                    id = int(id)
                    m = all_model_names[id]                    
                except Exception as e:
                    print(e)
                    result.append(None)
                    print(f'skipping {m}')

            stl_name = str(id) + '.stl'
            self.extract(stl_name, self.temp_dir)
            if dest_dir is None:
                result.append(o3d.io.read_triangle_mesh(os.path.join(self.temp_dir, stl_name)))
            else:
                shutil.move(os.path.join(self.temp_dir, stl_name), dest)
                print(f'{m} found and copied')
        return result


    def landmarks(self, translate=True, return_unknown=False):
        with self.open('Measure_Data_Info_new.bin') as f:
            t = f.read().decode("utf-8").strip(';').split(';')
            lmk = [x.split(',') for x in t]
            lmk = {int(l[0]):tuple(map(float,l[-3:])) for l in lmk}
            assert len(lmk)==len(t)

        if not translate:
            return lmk

        index = {}
        if os.path.exists('cass_landmark_index.json'):
            with open('cass_landmark_index.json','r') as f:
                index = json.loads(f.read())
                index = {int(k):v for k,v in index.items()}

        keys = list(lmk.keys())
        unknown = {}
        zeros = {}
        for l in keys:
            if not any(lmk[l]):
                zeros[l] = lmk.pop(l)
                continue
            if l in index:
                lmk[index[l]] = lmk.pop(l)
                # print(f'{l:>10}    -> {index[l]:>10}   {lmk_copy[index[l]]}') # for debugging
            else:
                unknown[l] = lmk.pop(l)
        
        # print stats
        print(f'{len(lmk):>5}    known landmarks')
        if len(zeros):
            print(f'{len(zeros):>5}    zeros\n  {zeros}')
        if len(unknown):
            print(f'{len(unknown):>5}    unknown indices\n           {list(unknown.keys())}')

        if return_unknown:
            unknown.update(zeros)
            return lmk, unknown
        else:
            return lmk
            

    # for maintaining a dictionary of landmark indices found in CASS files
    # different versions of database result in different dictionaries
    def update_landmark_index(self, xlsx, dictionary): # current_index: index:label
        index = {}
        if os.path.exists(dictionary):
            with open(dictionary,'r') as f:
                index = json.loads(f.read())
                index = {int(k):v for k,v in index.items()}

        lmk = self.landmarks(translate=False)

        xlsx = pandas.read_excel(xlsx, header=0).values
        xlsx[1:,1:] += xlsx[0:1,1:] # landmark offset
        lmk183 = {l[0]:l[1:].tolist() for l in xlsx[1:]}

        isequal = lambda x,y: bool(x*y) and round(x,1)==round(y,1)

        for l1,k1 in lmk.items():
            for l2,k2 in lmk183.items():
                if all(map(isequal, k1, k2)):
                    if l1 in index:
                        print(f'{l1:>5}    known to be {index[l1]:>10}')
                        assert index[l1] == l2, f'MISMATCH: {l1} -> {l2}'
                    else:
                        print(f'{l1:>5}    found to be {l2:>10}')
                        index[l1] = l2

        print(f'{len(index):>5}    known indices')
        
        with open('cass_landmark_index.json','w') as f:
            f.write(json.dumps(index))

        return index
                        