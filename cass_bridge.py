import rarfile, shutil, os, json, struct, datetime, csv
from collections import namedtuple
import numpy as np
from vtk_bridge import *
from stl import *

os.environ["PATH"] += os.pathsep + r'C:\Program Files\WinRAR'

class CASS(rarfile.RarFile):

    @property
    def db(self):
        if hasattr(self, '_db'):
            return self._db
        else:
            return None


    @db.setter
    def db(self, _db):
        self._db = _db


    @property
    def subject_info(self):
        '''reads the Sub_info(name, dob, and sex) from cass file
        executes only once per opened file'''

        # a convenience tuple
        Sub_info = namedtuple('Sub_info', ('name','dob','sex'))

        if not hasattr(self, '_subject_info'):
            with self.open(r'Patient_info.bin') as f:
                info = f.read().decode("utf-8").split(',')
                setattr(self, '_subject_info', Sub_info(*info[:3]))
        return self._subject_info


    @property
    def model_info(self):
        '''reads from cass file to return a list of model info dict's,
        which contains "name", "T" (transformations), and other unknown info
        executes only once per opened file'''

        # a convenience tuple
        Sub_info = namedtuple('Sub_info', ('name','T','sup'))

        if not hasattr(self, '_model_info'):
            with self.open('Three_Data_Info.bin') as f:
                t = f.read().decode("utf-8")

            # format into list with length equal to number of models contained in cass
            # this number should equal the number at the very beginning of above file
            model_info = [x.split(',') for x in t.strip(';').split(';')[1:]]

            for i,info in enumerate(model_info):
                # i is also id for each model
                name = info[0]
                T = np.array(info[41:201]+info[203:], dtype=float).reshape(-1,4,4)
                if T.shape[0] != 12:
                    print('total number of transformations mismatch')
                    T = []
                model_info[i] = Sub_info(name, T, info[1:41]+info[201:203])

            setattr(self, '_model_info', model_info)

        return self._model_info


    def write_transformation(self, model_name, dest_file):
        if not self.has_model(model_name):
            return None
        
        ind = self.model_indices([model_name])[0]
        with open(dest_file, 'w', newline='') as f:
            csv.writer(f, delimiter=' ').writerows(self.model_info[ind].T[0])

        return None



    @property
    def model_list(self):
        '''reads from cass file to return a list of model names'''
        if not hasattr(self, '_model_list'):
            setattr(self, '_model_list', [info.name for info in self.model_info])
        return self._model_list


    def print_model_list(self):
        for i,n in enumerate(self.model_list):
            print(f'{i:>5} - {n}')
        return None


    def has_model(self, model_name):
        return model_name in self.model_list


    def model_indices(self, model_list):
        '''returns internal index of each specified model
        returns None is model is not found
        "{inidex}.stl" is also the name of the model file in cass'''
        
        # create list for return
        indices = [None]*len(model_list)

        for i,name in enumerate(model_list):
            if self.has_model(name):
                indices[i] = self.model_list.index(name)

        return indices


    def copy_models(self, model_list, dest_dir, transform=True, fail_if_exists=False, user_select=False):
        '''copies specified models to specified destination directory
        does not return or err
        does log for failure'''

        # prepare temporary and destination directories
        # use temporary to prevent accidental override
        dest_dir = os.path.normpath(dest_dir)
        temp_dir = os.path.join(
            os.path.dirname(dest_dir),
            '~' + os.path.basename(dest_dir) + datetime.datetime.now().strftime('%y%M%d%H%m%S')
        )
        os.makedirs(temp_dir)
        os.makedirs(dest_dir, exist_ok=True)

        # model indices are also stl file names to copy
        model_indices = self.model_indices(model_list)

        for id, mn in zip(model_indices, model_list):

            # file destination
            dest = os.path.join(dest_dir, mn+'.stl')
            
            # if model already exists and override is not allowed
            if os.path.exists(dest) and fail_if_exists:

                # fail this copying
                print(f'copy failed, {dest} exists')
                break

            # if model does not exist in cass
            if id is None:

                # select other model to copy, but into the specified model name
                # this deals with mis-named models
                if user_select: 
                    id = self.user_select_model(mn)

                    # user gave invalid input
                    if id is None: 
                        break
                
                # cass file does not contain specified model, fail this copying
                else:
                    print(f'copy failed, {mn} not found')
                    break

            # copy model from cass to temporary dir
            self.extract(f'{id}.stl', temp_dir)

        else: # if all models are present

            # move files from temporary dir to dest dir
            for id, mn in zip(model_indices, model_list):
                if transform:
                    # transform stl to its most recent position and replace the existing file
                    m = read_stl(os.path.join(temp_dir, f'{id}.stl'))
                    transform_stl(m, self.model_info[id].T[0])
                    write_stl(m, os.path.join(temp_dir, f'{id}.stl'))

                shutil.move(os.path.join(temp_dir, f'{id}.stl'), os.path.join(dest_dir, mn+'.stl'))

        # regardless of success, clean up and return
        shutil.rmtree(temp_dir)
        return None


    def user_select_model(self, model_name, retry=True):
        '''allows user to interactively select model by typing index into console'''

        self.print_model_list()

        while 1:
            # take user input
            model_id = input(f'type model index to select model "{model_name}": ')

            # validate input and return
            try:
                model_id = int(model_id)
                print(f'\r{id:>5} - {self.model_list[model_id]} selected')
                return model_id
            except:
                print(f'invalid input: {model_id}')
                if not retry:
                    return None


    def load_models(self, model_list):
        '''loads specified model into Model(v,f,fn,T) tuple
        where T is a list of transformation matrices at various stages of the planning'''

        # model indices are also stl file names to copy
        indices = self.model_indices(model_list)

        # create list for return
        models = [None]*len(model_list)

        # define a convenience tuple
        Model = namedtuple('Model',('v','f','fn','T'))

        for i,id in enumerate(indices):
            
            # cass file does not contain such model
            if id is None:
                continue

            # load stl model
            models[i] = Model(
                *read_stl(f'{id}.stl'), # the v,f,fn
                self.model_info[id].T # the T
            )

        return models


    def load_landmarks(self):
        '''loads landmarks contained in cass file
        if self.db is set, uses it to interpret landmark labels
        if not, returns id as label'''
        
        # open binary text file
        with self.open('Measure_Data_Info_new.bin') as f:

            # read and decode binary text file
            t = f.read().decode("utf-8")

        # to accomodate old landmark format in cass
        old = False
        if 'No' in t:
            old = True

        # format into id:coordinates
        lmk = [x.split(',') for x in t.strip(';').split(';')]
        lmk = {int(l[0]):tuple(map(float,l[-4:-1] if old else l[-3:])) for l in lmk}
        
        # if db is set, interpret into {label:coordinates} dict
        if self.db: 
            lmk_interpreted = {}
            for k,v in lmk.items():

                # find landmark label for current id
                x = self.db.find(ID=k)

                # record landmarks only if label is known in db
                if x is not None: 
                    lmk_interpreted[x.Name] = v

            return lmk_interpreted
        
        # if db is not set, return {id:coordinates} dict
        else:
            return lmk

    def write_landmarks(self, lmk_file):
        lmk = self.load_landmarks()
        with open(lmk_file, 'w', newline='') as f:
            csv.writer(f).writerows([[k] + list(v) for k,v in lmk.items()])
        
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