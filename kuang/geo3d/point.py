from collections.abc import Sequence

import numpy as np
from .transform import Transform, Transformable
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkDoubleArray, vtkPoints

class PointArray(np.ndarray, Transformable):

    def __new__(cls, data=()):
        if isinstance(data, vtkPoints):
            arr = np.empty((data.GetNumberOfPoints(),3), dtype=np.float64).view(cls)
            arr.nx3 = data
        else:
            arr = np.array(data, dtype=float).reshape(-1,3).view(cls)
        return arr
    
    @property
    def nx3(self):
        return self

    @nx3.setter
    def nx3(self, nx3):
        if isinstance(nx3, Sequence):
            nx3 = np.array(nx3)
        if isinstance(nx3, np.ndarray):
            self[...] = nx3
        elif isinstance(nx3, vtkPoints):
            d = nx3.GetData()
            if d.GetNumberOfTuples() != len(self):
                raise ValueError('size mismatch')
            if isinstance(d, vtkFloatArray):
                self[...] = np.frombuffer(d, dtype=np.float32).reshape(-1,3)
            elif isinstance(d, vtkDoubleArray):
                self[...] = np.frombuffer(d, dtype=np.float64).reshape(-1,3)
            else:
                raise ValueError('can only set nx3 with vtk float or data array')

        else:
            raise ValueError('cannot set nx3')

    # coordinates getter returns a COPY of nx3
    # the returned value should always be safe
    @property
    def coordinates(self):
        return self.nx3.view(PointArray).copy()

    @coordinates.setter
    def coordinates(self, arr: np.ndarray) -> None:
        self.nx3 = arr
        return None

    @property
    def centroid(self):
        return self.nx3.mean(axis=0, keepdims=True)

    def transform(self, T:Transform) -> None:
        if T.pre_multiply:
            T = T.T
        self.nx4 = self.nx4@T
        return None

    def translate(self, t) -> None:
        self.nx3 = self.nx3 + np.array(t).reshape(1,3)
        return None
    
    def rotate(self, R, pre_multiply=False, center=None) -> None: 
        R = np.asarray(R)
        T = Transform(pre_multiply=pre_multiply)
        if center is not None:
            c = np.array(center)
            T.translate(-c)
            T.rotate(R)
            T.translate(c)
        else:
            T.rotate(R)
        self.transform(T)
        return None

    def scale(self, s: float, center=None) -> None:
        self.nx3 = self.nx3 * s
        if center is not None:
            self.translate(np.array(center) * (1-s))
        return None


class NamedArray(PointArray):

    dtype = np.dtype([('name', str, 12), ('coordinates', float, (3,))])

    def __new__(cls, *name_coord_pairs, dtype=None):
        '''usage:
            1. namedarray( object )
                object can be zipped name and coordinates pairs, dict of name:coordinates, existing namedarrays
            2. namedarray( (n1,p1), (n2,p2), ...)
        '''        
        if len(name_coord_pairs) == 1:
            arr = name_coord_pairs[0]
            if isinstance(arr, zip):
                name_coord_pairs = list(arr)
            elif isinstance(arr, dict):
                name_coord_pairs = list(arr.items())
            elif isinstance(arr, cls):
                return cls(arr)
        if dtype is None:
            dtype = cls.dtype
        return np.array(list(name_coord_pairs), dtype=dtype).view(cls)
    
    def append(self, *name_coord_pairs):
        # same func signature as __new__
        arr_append = self.__class__(*name_coord_pairs, dtype=self.dtype)
        return np.append(self, arr_append).view(self.__class__)
    
    @property
    def nx3(self):
        return self['coordinates'].view(np.ndarray)

    @nx3.setter
    def nx3(self, nx3: np.ndarray):
        self['coordinates'].view(PointArray).nx3 = nx3


def test():
    p = vtkPoints()
    p.InsertNextPoint(0,1,2)
    p.InsertNextPoint(1,2,3)
    x = NamedArray(zip(('x','y'), np.zeros((2,3))))
    print(x)
    x.nx3 = p
    x.nx3[:,1] = 2
    print(x)
    x.nx3 = p
    print(x)
    x.nx3 = ((1,2,3),(2,3,4))
    print(x)
    # R = np.random.rand(3,3)
    # c = np.random.rand(3,).tolist()
    # t = np.random.rand(3,).tolist()
    # P = PointArray(np.random.rand(10,3))
    # PP = NamedArray(zip('abcdefghij',P))
    # premul = True
    # def test(x):
    #     x.translate(t)
    #     x.rotate(R, pre_multiply=premul, center=c)

    # test(P)
    # T = Transform(pre_multiply=premul)
    # test(T)

    # print(P)
    # PP.transform(T)
    # print(PP)
    # print(len(PP[2]['name']), type(PP[2]['name']), isinstance(PP[2]['name'], str), isinstance(PP[2]['coordinates'][2], float))
    # print(np.all(np.isclose(P.nx3,PP.nx3)))

    # PPnew = PP.append(('x',(1,2,3)))
    # print(PPnew)
    # print(PPnew.dtype)
    # k = NamedArray().append(('x',(1,2,3)),('y',(4,5,6)))
    # print(type(k))
    # print(k.dtype)    


if __name__=='__main__':
    test()
