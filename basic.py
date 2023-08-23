import numpy as np
import open3d as o3d
import pyvista
from time import perf_counter as toc
__ALL__ = ["Transform", "Plane", "Pointset", "Poly"]


class Transform(np.ndarray):
    # Similarity transformation only
    def __new__(cls, *shape):
        # Transform(2,3) creates id transformation matrix of size (2,3,4,4)
        # this shape setup is compatible with many numpy matrix functions
        return np.tile(np.eye(4), (*shape, 1, 1)).view(cls)

    # use following function with matrix input and do not reduce dimensionality
    def translate(self, t):
        self[..., 3:, :3] += self[..., 3:, 3:] * t
        return self

    def rotate(self, R):
        self[..., :, :3] = self[..., :, :3] @ R
        return self

    def scale(self, s):
        self[..., 3:, 3:] /= s
        return self
        
    def reflect(self, plane:"Plane"):
        plane = np.asarray(plane).flatten()
        T = np.eye(4) - 2*plane[:,None]@[[*plane[:-1],0]]/(plane[:-1].dot(plane[:-1])) # such that T@T === eye(4)
        # plane = plane[...,None,:]
        # nTn = plane[...,None] @ plane[...,None,:]

        self[...] = self @ T
        return self


class Plane(np.ndarray):
    # self = [a,b,c,d] s.t. ax + by + cz + d = 0
    # vector abc is normalized
    # n*p + d = 0
    @property
    def d(self):
        return self[3]

    @property
    def n(self):
        return self[:3]

    @property
    def p(self): # closest point to origin on plane
        return Pointset(-self.n*self.d)

    def __new__(cls, *args, **kwargs):
        abcd = cls.__bases__[0].__new__(*args, **kwargs)
        abcd = abcd.flatten()/np.sum(abcd.flat[:3]**2)**.5
        return abcd.view(cls)

    def reset(self, **kwargs):
        if 'equation' in kwargs:
            self[:] = kwargs['equation'].flatten()
        elif 'normal' in kwargs:
            if 'd' in kwargs:
                pass
            # not finished

    @classmethod
    def pca_fit(cls, points: "Pointset"):
        cen = points.centroid
        points = points - cen
        _, _, W = np.linalg.svd(points.T@points)
        pln = cls.get(normal=W[-1].flatten(), point=cen.flatten())
        return pln

    @classmethod
    def get(cls, normal, point):
        normal = normal/np.sum(normal**2)**.5
        return np.asarray([*normal, -normal.dot(point)]).view(cls)

    def reflect(self, points: "Pointset"):
        points = points - self.distance(points) @ self.n[None,:] * 2
        return points

    def distance(self, points: "Pointset"):
        return self.d + points.dot(self.n).reshape([-1,1])

    def segment(self, poly: "Poly"):
        if isinstance(poly, Poly):
            d = self.distance(poly.V)
            return \
                Poly(V=poly.V,F=poly.F[np.isin(poly.F,(d>0).nonzero()[0]).all(axis=1)]), \
                Poly(V=poly.V,F=poly.F[np.isin(poly.F,(d<0).nonzero()[0]).all(axis=1)])
        else:
            d = self.distance(Pointset(poly.vertices))
            poly.vertex_colors = o3d.utility.Vector3dVector((d>0)@[[.2,.5,.5]] + (d<0)@[[.5,.2,.2]])
            return poly



class Pointset(np.ndarray):

    def __new__(cls, arr):
        return np.asarray(arr).reshape(-1,3).view(cls)

    @property
    def centroid(self):
        return self.mean(axis=self.ndim-2, keepdims=True)

    def setdiff(self, other:'Pointset'):
        this, other = self.tolist(), other.tolist()
        return self.__class__([x for x in this if x not in other])

    def transform(self, T):
        self[...] = (self @ T[...,:3,:3] + T[...,3:,:3]) / T[...,3:,3:]
        return self

    @property
    def quad1(self):
        ps = np.empty((self.shape[0], self.shape[1]+1))
        ps[:,:3], ps[:,3] = self, 1
        return ps

    @property
    def quad0(self):
        ps = np.empty((self.shape[0], self.shape[1]+1))
        ps[:,:3], ps[:,3] = self, 0
        return ps


class Poly(pyvista.PolyData):

    @classmethod
    def read(cls, file):
        return cls(pyvista.get_reader(file).read())

    def __init__(self, *args, **kwargs):
        # when using key words 'V' and 'F', use custom init
        # otherwise use PolyData's
        vtk_faces = lambda f: np.hstack((np.tile(3,(f.shape[0],1)), f.astype(int))).flatten()
        if len(args) and isinstance(args[0],dict) and 'V' in args[0] and 'F' in args[0]:
            super().__init__(np.asarray(args[0]['V']), vtk_faces(np.asarray(args[0]['F'])))
        elif 'V' in kwargs and 'F' in kwargs:
            super().__init__(np.asarray(kwargs['V']), vtk_faces(np.asarray(kwargs['F'])))
        else:
            super().__init__( *args, **kwargs )
        self.last_modified = dict(V=toc())

    def update(self, attr, v): # private method to update properties of the class
        if attr not in self.last_modified or self.last_modified[attr] < self.last_modified['V']:
            setattr(self, '_'+attr, v)
            self.last_modified[attr] = toc()

    @property # vertices
    def V(self):
        return self.points.view(Pointset)

    @V.setter
    def V(self, v):
        assert self.points.size == v.size
        self.points[...] = v
        self.last_modified['V'] = toc()

    @property # faces
    def F(self):
        return self.faces.reshape(-1,4)[:,1:]

    @property # vertex indicex of free/boundary points
    def E(self): # does not update with vertices since our program does not change faces
        if not hasattr(self, '_E'):
            edg_all = np.sort(self.F[:,[0,1,1,2,2,0]].reshape(-1,2), axis=1)
            edg, ind = np.unique(edg_all, axis=0, return_inverse=True)
            edg = np.unique(edg[np.bincount(ind)==1])
            setattr(self, '_E', edg)
        return self._E


    @property # vertex normal
    def VN(self):
        self.update('VN', self.point_normals)
        return self._VN

    @property # face normal
    def FN(self):
        self.update('FN', self.face_normals)
        return self._FN

    @property # curvature
    def C(self):
        self.update('C', self.curvature())
        return self._C

    @property
    def mean_edge_len(self):
        self.update(
            'mean_edge_len', 
            np.mean(((self.V[self.F[:,[0,1,2]],:]-self.V[self.F[:,[1,2,0]],:])**2).sum(axis=-1)**.5)
        )
        return self._mean_edge_len

    def to_o3d(self):
        return o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(self.V),
                triangles=o3d.utility.Vector3iVector(self.F)
            ).compute_vertex_normals()


def regular_icosahedron():
    phi = .5 + .5 * 5 ** .5
    V = np.asarray([
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1)], dtype=float)
    V = V / np.sum(V**2, axis=1)[:,None]**.5
    F = np.asarray([
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1)], dtype=int)

    return Poly(V=V, F=F)


def sphere(radius=1., center=(0,0,0), edge_len=None):
    # subdivide a regular icosahedron iteratively until
    # edge length < edge_len * radius
    center = np.array(center).reshape(-1,3)
    edge_len = .3 if edge_len is None else edge_len/radius
    s = regular_icosahedron()
    V,F = s.V, s.F
    while np.sum((V[F[0,0]] - V[F[0,1]])**2)**.5 > edge_len:
        nv, nf = len(V), len(F)

        V0, V1, V2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
        V = np.vstack((V, V0/2+V1/2, V1/2+V2/2, V0/2+V2/2))
        V = V / np.sum(V**2, axis=1, keepdims=True)**.5

        ind01, ind12, ind02 = np.arange(nv,nv+nf), np.arange(nv+nf,nv+nf*2), np.arange(nv+nf*2,nv+nf*3)
        f0 = np.vstack((ind02,F[:,0],ind01))
        f1 = np.vstack((ind01,F[:,1],ind12))
        f2 = np.vstack((ind12,F[:,2],ind02))
        fc = np.vstack((ind02,ind01,ind12))
        F = np.hstack((f0,f1,f2,fc)).T

    sph = Poly()
    for c in center:
        sph.merge(Poly(V=V*radius+c, F=F), merge_points=False, inplace=True)
    return sph.clean()

