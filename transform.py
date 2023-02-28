from abc import ABC, abstractmethod
import numpy as np

class Transform3D(np.ndarray):
    # transformation matrix for geometric objects
    # similarity transformation only

    pre_multiply=False # set this class property based on each project's need

    def __new__(cls, **kwargs):
        T = np.eye(4,4).view(cls)
        if 'pre_multiply' in kwargs and kwargs['pre_multiply']:
            T.pre_multiply = True
        return T

    def transform(self, T: "Transform3D"):
        if T.pre_multiply != self.pre_multiply:
            T = T.T
        self[...] = T@self if self.pre_multiply else self@T
        return None

    def translate(self, t):
        t = np.asarray(t)
        if self.pre_multiply:
            self[:3,3] += self[3,3] * t.flat
        else:
            self[3,:3] += self[3,3] * t.flat
        return None

    def rotate(self, R, **kwargs):
        # BE MINDFUL
        # R should be compatible with self.pre_multiply
        # if not, simply take transpose (R.T) before assignment
        R = np.asarray(R)
        if 'center' in kwargs:
            c = np.array(kwargs['center'])
        else:
            c = np.array((0.,0.,0.))
        self.translate(-c)
        if self.pre_multiply:
            self[:3,:] = R @ self[:3,:]
        else:
            self[:,:3] = self[:,:3] @ R
        self.translate(c)
        return None

    def scale(self, s):
        self[3, 3] = self[3, 3] / s
        return None
        
class Transformable3D(ABC):

    # IMPLEMENT THESE ABSTRACT METHOD
    @property
    @abstractmethod
    # expose coordinates to calculation
    # avoid unintended change of the returned array
    # safest way is to copy coordinates into new array
    def nx3(self): pass 

    @nx3.setter
    @abstractmethod
    # set coordinates after calculation
    # alters self
    def nx3(self, nx3 : np.ndarray): pass # to set coordinates

    @property
    def nx4(self): # convenience getter
        return np.hstack((self.nx3, np.ones((self.shape[0],1))))

    @nx4.setter
    def nx4(self, nx4 : np.ndarray): # convenience setter
        self.nx3 = nx4[:,:3]
        return None

    @property
    def centroid(self):
        return self.nx3.mean(axis=0, keepdims=True)

    def transform(self, T: Transform3D):
        if T.pre_multiply:
            T = T.T
        self.nx4 = self.nx4@T
        return None

    def translate(self, t):
        self.nx3 = self.nx3 + np.array(t).reshape(1,3)
        return None
    
    def rotate(self, R, pre_multiply=False, **kwargs): 
        R = np.asarray(R)
        T = Transform3D(pre_multiply=pre_multiply)
        if 'center' in kwargs:
            c = np.array(kwargs['center'])
            T.translate(-c)
            T.rotate(R)
            T.translate(c)
        else:
            T.rotate(R)
        self.transform(T)
        return None

    def scale(self, s: float, **kwargs):
        self.nx3 = self.nx3 * s
        if 'center' in kwargs:
            self.translate(np.array(kwargs['center']) * (1-s))
        return None

