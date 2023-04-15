import rarfile, shutil, os, json, pandas
import open3d as o3d
from datetime import datetime as dt


class CASS(rarfile.RarFile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setattr(self, 'temp_dir', rf'c:\_temp_cass\{dt.now().__str__()}'.replace(' ','-').replace(':','').split('.',1)[0])
        os.makedirs(self.temp_dir, exist_ok=False)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, typ, value, traceback):
        self.close() 
        return super().__exit__(typ, value, traceback) # super's close() called herein

    def close(self, *args, **kwargs):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @property
    def model_names(self):
        if not hasattr(self, '_model_names'):
            with self.open('Three_Data_Info.bin') as f:
                t = f.read().decode("utf-8").strip(';').split(';')
                names = [x.split(',')[0] for x in t[1:]]
                assert t[0] == str(len(names)), 'total number of models mismatch, possibly due to file corruption'
                setattr(self, '_model_names', names)
        return self._model_names

    def print_model_names(self):
        for i,n in enumerate(self.model_names):
            print(f'{i:>5} - {n}')

    def model_index(self, name_list):
        names = self.model_names
        idx_list = [names.index(x) if x in names else -1 for x in names]
        return idx_list

    def select_model(self, model_name):
        self.print_model_names()
        id = input(f'type index to select model: {model_name}')
        try:
            id = int(id)
            print(f'\r{id:>5} - {self.model_names[id]} selected')
        except:
            raise ValueError('invalid input')
        assert id < len(self.model_names), 'outside the range'
        return id

    def copy_model(self, model_names, dest_dir, override=True, user_input=False):
        dest_dir = os.path.normpath(dest_dir)
        temp_dir = os.path.join(
            os.path.dirname(dest_dir),
            '~'+os.path.basename(dest_dir)
        )
        os.makedirs(dest_dir, exist_ok=True)
        model_indices = self.model_index(model_names)
        try:
            os.makedirs(temp_dir)
            for id, mn in zip(model_indices, model_names):
                if id<0:
                    if user_input:
                        id = self.select_model
                    self.extract(f'{id}.stl', temp_dir)
                    dest = os.path.join(dest_dir, mn+'.stl')
                    shutil.move(os.path.join(temp_dir, f'{id}.stl'), dest)
                else:
                    if user_input:


        
        if not override and os.path.isfile(dest):
            print(f'{dest} exists, skipped')
        self.extract(stl_name, temp_dir)
        shutil.move(os.path.join(temp_dir, stl_name), dest)


    def landmarks(self, dictionary='cass_landmark_index_nct.json', return_unknown=False):
        with self.open('Measure_Data_Info_new.bin') as f:
            t = f.read().decode("utf-8").strip(';').split(';')
            lmk = [x.split(',') for x in t]
            lmk = {int(l[0]):tuple(map(float,l[-3:])) for l in lmk}
            assert len(lmk)==len(t)

        if not dictionary:
            return lmk

        index = {}
        with open(dictionary,'r') as f:
            index = json.loads(f.read())
            index = {int(k):v for k,v in index.items()}

        keys = list(lmk.keys())
        unknown = {}
        zeros = {}
        for k in keys:
            if not any(lmk[k]):
                zeros[k] = lmk.pop(k)
                continue
            if k in index:
                lmk[index[k]] = lmk.pop(k)
                # print(f'{l:>10}    -> {index[l]:>10}   {lmk_copy[index[l]]}') # for debugging
            else:
                unknown[k] = lmk.pop(k)
        
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
    # check all landmarks coordinates are correct in AnatomicAligner
    # run this function with xlsx file exported from AnatomicAligner
    def update_landmark_index(self, xlsx_file, dictionary):
        index = {}
        if os.path.exists(dictionary):
            with open(dictionary,'r') as f:
                index = json.loads(f.read())
                index = {int(k):v for k,v in index.items()}

        lmk = self.landmarks(dictionary='')

        xlsx = pandas.read_excel(xlsx_file, header=0).values
        xlsx[1:,1:] += xlsx[0:1,1:] # landmark offset
        lmkxlsx = {l[0]:l[1:].tolist() for l in xlsx[1:]}

        isequal = lambda x,y: bool(x*y) and round(x,1)==round(y,1)

        for l1,k1 in lmk.items():
            for l2,k2 in lmkxlsx.items():
                if all(map(isequal, k1, k2)):
                    if l1 in index:
                        print(f'{l1:>5}    known to be {index[l1]:>10}')
                        assert index[l1] == l2, f'MISMATCH: {l1} -> {l2}'
                    else:
                        print(f'{l1:>5}    found to be {l2:>10}')
                        index[l1] = l2

        print(f'{len(index):>5}    known indices')
        
        with open(dictionary,'w') as f:
            f.write(json.dumps(index))

        return index
                        