from transform import Transformable3D, Transform3D
import numpy as np

class pointarray(np.ndarray, Transformable3D):

    def __new__(cls, arr):
        return np.array(arr).reshape(-1,3).view(cls)
    
    @property
    def nx3(self):
        return np.array(self)

    @nx3.setter
    def nx3(self, nx3 : np.ndarray):
        self[...] = nx3


class namedarray(np.ndarray, Transformable3D):

    dtype = np.dtype([('name', str, 16), ('coordinates', float, (3,))])

    def __new__(cls, name_coord_zipped=[]):
        return np.array(name_coord_zipped, dtype=cls.dtype).view(cls)
    
    @property
    def nx3(self):
        return pointarray(self['coordinates'])

    @nx3.setter
    def nx3(self, nx3 : pointarray):
        self['coordinates'] = nx3

    def append(self, *name_coord_tuple):
        arr_append = np.array(list(name_coord_tuple), dtype=self.dtype).view(self.__class__)
        new_namedarray = np.append(self, arr_append)
        return new_namedarray



if __name__=='__main__':
    t = np.random.rand(3,)
    R = np.random.rand(3,3)
    c = np.random.rand(3,).tolist()
    P = pointarray(np.random.rand(10,3))
    PP = namedarray([*zip(iter('abcdefghij'),P)])
    premul = True
    def test(x):
        x.translate(t)
        x.rotate(R, pre_multiply=premul, center=c)

    test(P)
    T = Transform3D(pre_multiply=premul)
    test(T)

    print(P)
    PP.transform(T)
    print(PP)
    print(len(PP[2]['name']), type(PP[2]['name']), isinstance(PP[2]['name'], str), isinstance(PP[2]['coordinates'][2], float))
    print(np.all(np.isclose(P.nx3,PP.nx3)))

    PPnew = PP.append(('x',(1,2,3)))
    print(PPnew)
    print(PPnew.dtype)
    k = namedarray().append(('x',(1,2,3)),('y',(4,5,6)))
    print(k.dtype)