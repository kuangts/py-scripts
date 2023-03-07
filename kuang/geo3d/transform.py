from abc import ABC, abstractmethod
import numpy as np

# add pitch roll yaw

class Transform(np.ndarray):
    # transformation matrix for geometric objects
    # similarity transformation only

    pre_multiply=False # set this class property based on each project's need

    def __new__(cls, **kwargs):
        T = np.eye(4,4).view(cls)
        if 'pre_multiply' in kwargs and kwargs['pre_multiply']:
            T.pre_multiply = True
        return T

    def transform(self, T: "Transform"):
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
        

class Transformable(ABC):

    # IMPLEMENT THESE ABSTRACT METHOD
    @property
    @abstractmethod
    # expose coordinates to calculation
    # avoid unintended change of the returned array
    # safest way is to copy coordinates into new array
    # mainly for internal use
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
    
    @abstractmethod
    def transform(self, T:Transform) -> None: pass

    @abstractmethod
    def translate(self, t) -> None: pass
    
    @abstractmethod
    def rotate(self, R, pre_multiply=False, center=None, **kwargs) -> None:  pass

    @abstractmethod
    def scale(self, s: float, **kwargs) -> None: pass

