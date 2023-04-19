import rarfile, shutil, os, json, struct
from collections import namedtuple
import numpy as np
from vtk_bridge import *

os.environ["PATH"] += os.pathsep + r'C:\Program Files\WinRAR'


class CASS(rarfile.RarFile):

    @property
    def subject_info(self):
        if not hasattr(self, '_subject_info'):
            with self.open(r'Patient_info.bin') as f:
                setattr(self, '_subject_info', f.read().decode("utf-8").split(',')[:3])
        return self._subject_info


    @property
    def model_info(self):
        if not hasattr(self, '_model_info'):
            with self.open('Three_Data_Info.bin') as f:
                t = f.read().decode("utf-8").strip(';').split(';')
                model_info = [x.split(',') for x in t[1:]]
            for i,info in enumerate(model_info):
                name = info[0]
                T = np.array(info[41:201]+info[203:], dtype=float).reshape(-1,4,4)
                if T.shape[0] != 12:
                    print('total number of transformations mismatch')
                    return []
                sup0 = info[1:41]
                sup1 = info[201:203]
                model_info[i] = dict(name=name, T=T.tolist(), sup0=sup0, sup1=sup1)
            setattr(self, '_model_info', model_info)
        return self._model_info


    @property
    def model_names(self):
        if not hasattr(self, '_model_names'):
            setattr(self, '_model_names', [info['name'] for info in self.model_info])
        return self._model_names

    def has_model(self, model_name):
        return model_name in self.model_names

    def print_model_names(self):
        for i,n in enumerate(self.model_names):
            print(f'{i:>5} - {n}')

    def model_indices(self, model_names):
        model_indices = [None]*len(model_names)
        for i,name in enumerate(model_names):
            if name in self.model_names:
                model_indices[i] = self.model_names.index(name)
        return model_indices

    def copy_models(self, model_names, dest_dir, override=True, user_input=False):
        dest_dir = os.path.normpath(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        temp_dir = os.path.join(
            os.path.dirname(dest_dir),
            '~'+os.path.basename(dest_dir)
        )
        while os.path.exists(temp_dir):
            temp_dir += '-1'
        else:
            os.makedirs(temp_dir)            
        model_indices = self.model_indices(model_names)

        for id, mn in zip(model_indices, model_names):
            dest = os.path.join(dest_dir, mn+'.stl')
            if not override and os.path.exists(dest):
                print(f'{dest} exists, skipped')
                continue
            if id is None:
                if user_input:
                    id = self.select_model(mn)
                    if id is None:
                        print(f'{mn} not found, skipped')
                        continue
                else:
                    print(f'{mn} not found, skipped')
                    continue
            self.extract(f'{id}.stl', temp_dir)
            shutil.move(os.path.join(temp_dir, f'{id}.stl'), dest)
        
        shutil.rmtree(temp_dir)
        # os.rmdir(temp_dir)
        return None


    def select_model(self, model_name):
        self.print_model_names()
        model_id = input(f'type index to select model "{model_name}": ')
        try:
            model_id = int(model_id)
            print(f'\r{id:>5} - {self.model_names[model_id]} selected')
            return id
        except:
            print(f'invalid input {model_id}')
            return None


    def load_models(self, model_names):
        indices = self.model_indices(model_names)
        models = [None]*len(model_names)
        Model = namedtuple('Model',('v','f','fn','T'))
        for i,id in enumerate(indices):
            if id is None:
                continue
            info = self.model_info[id]
            with self.open(f'{id}.stl') as f:
                f.seek(80)
                data = f.read()
            nf, data = struct.unpack('I', data[0:4])[0], data[4:]
            data = struct.unpack('f'*(nf*12), b''.join([data[i*50:i*50+48] for i in range(nf)]))
            data = np.asarray(data).reshape(-1,12)
            FN = data[:,0:3].astype(np.float32)
            V = data[:,3:12].reshape(-1,3).astype(np.float32)
            F = np.arange(0,len(V)).reshape(-1,3).astype(np.int64)
            models[i] = Model(v=V, f=F, fn=FN, T=info['T'])
        return models


    def load_landmarks(self, interpreter=None):
        with self.open('Measure_Data_Info_new.bin') as f:
            t = f.read().decode("utf-8").strip(';').split(';')
        lmk = [x.split(',') for x in t]
        lmk = {int(l[0]):tuple(map(float,l[-3:])) for l in lmk}
        
        if interpreter is not None:
            lmk_new = {}
            lmk_del = {}
            for k,v in lmk.items():
                x = interpreter.find(ID=k)
                if x is not None:
                    lmk_new[x.Name] = v
                else:
                    lmk_del[k] = v
            lmk = lmk_new

        return lmk


    def copy_landmarks(self, file_to_write):
        with self.open('Measure_Data_Info_new.bin') as f:
            t = f.read().decode("utf-8")
        with open(file_to_write, 'w') as f:
            f.write(t)

        return None




    # def landmarks(self, dictionary='cass_landmark_index_nct.json', return_unknown=False):
    #     with self.open('Measure_Data_Info_new.bin') as f:
    #         t = f.read().decode("utf-8").strip(';').split(';')
    #         lmk = [x.split(',') for x in t]
    #         lmk = {int(l[0]):tuple(map(float,l[-3:])) for l in lmk}
    #         assert len(lmk)==len(t)

    #     if not dictionary:
    #         return lmk

    #     index = {}
    #     with open(dictionary,'r') as f:
    #         index = json.loads(f.read())
    #         index = {int(k):v for k,v in index.items()}

    #     keys = list(lmk.keys())
    #     unknown = {}
    #     zeros = {}
    #     for k in keys:
    #         if not any(lmk[k]):
    #             zeros[k] = lmk.pop(k)
    #             continue
    #         if k in index:
    #             lmk[index[k]] = lmk.pop(k)
    #             # print(f'{l:>10}    -> {index[l]:>10}   {lmk_copy[index[l]]}') # for debugging
    #         else:
    #             unknown[k] = lmk.pop(k)
        
    #     # print stats
    #     print(f'{len(lmk):>5}    known landmarks')
    #     if len(zeros):
    #         print(f'{len(zeros):>5}    zeros\n  {zeros}')
    #     if len(unknown):
    #         print(f'{len(unknown):>5}    unknown indices\n           {list(unknown.keys())}')

    #     if return_unknown:
    #         unknown.update(zeros)
    #         return lmk, unknown
    #     else:
    #         return lmk
            

    # for maintaining a dictionary of landmark indices found in CASS files
    # different versions of database result in different dictionaries
    # check all landmarks coordinates are correct in AnatomicAligner
    # run this function with xlsx file exported from AnatomicAligner
    def update_landmark_index(self, xlsx_file, dictionary):
        import pandas
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
                        

def main(cass_file=r'kuang/tests/n09.CASS'):
    with CASS(cass_file) as f:
        models = f.load_models(['manu-maxi pre'])

if __name__=='__main__':
    main()