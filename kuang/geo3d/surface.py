#! /usr/bin/env python

import struct, copy, re, plotly, typing
from collections.abc import Sequence

import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
from vtkmodules.vtkCommonColor import vtkNamedColors

import vtk

from .point import PointArray

# to_vtk: IsStorage64Bit is not resolved


class _Base: 
	
	def copy(self):
		return copy.deepcopy(self)

	@classmethod
	def merge(cls, *args, **initargs):
		# does not alter input args
		s = cls()
		conn, pnts = s.connectivity, s.points
		for arg in args:
			assert isinstance(arg, cls), 'cannot merge obj from different class'
			conn = np.vstack((conn, arg.connectivity+pnts.shape[0])) 
			pnts = np.vstack((pnts, arg.points))
		
		return cls(pnts, conn, **initargs)

	def __init__(self, points, connectivity):
		assert ( not len(points)) == ( not len(connectivity)), 'must specify points and connectivity both, or neither'
		self.points = PointArray(points)
		self.connectivity = np.array(connectivity, dtype=int)

	def remove_duplicate(self, decimals=6):
		if not self.points.size or not self.connectivity.size:
			return
		self.points = self.points.round(decimals=decimals)
		self.points, ind = np.unique(self.points, axis=0, return_inverse=True)
		self.connectivity = np.unique(ind[self.connectivity], axis=0)
		# f_unique, ind = np.unique(self.connectivity, return_inverse=True)
		# self.connectivity = ind.reshape(self.connectivity.shape)
		# self.points = self.points[f_unique,:]
		return None

	@classmethod
	def read(cls, file, **initargs): # convenience method to keep consistent use of attr 'read', meant to be overriden
		return cls.read_npz(file, **initargs)

	def write(self, file, **write_dict):
		self.__class__.write_npz(file, points=self.points, connectivity=self.connectivity, **write_dict)

	def __repr__(self):
		s = f'points: {self.points.shape}\n{self.points}\n' + f'connectivity: {self.connectivity.shape}\n{self.connectivity}\n'
		return s
	
	@classmethod
	def read_npz(cls, file, point_property_name='points', conn_property_name='connectivity', *initargs): # do not override
		d = dict(np.load(file))
		return cls(d.pop(point_property_name), d.pop(conn_property_name), **d, **initargs)

	@classmethod
	def write_npz(self, file, **kwargs):
		np.savez(file, points=self.points, connectivity=self.connectivity, **kwargs)



class TriangleSurface(_Base):

	number_of_components = 3

	def __init__(self, vertices, faces, **initargs):
		super().__init__(vertices, faces)
		if 'remove_duplicate' in initargs and initargs['remove_duplicate']:
			self.remove_duplicate()


	@property
	def vertices(self):
		return self.points


	@property
	def faces(self):
		return self.connectivity


	@property
	def face_normals(self):
		v10 = self.vertices[self.faces[:,2],:] - self.vertices[self.faces[:,0],:]
		v20 = self.vertices[self.faces[:,-1],:] - self.vertices[self.faces[:,0],:]
		fn = np.cross(v10, v20)
		fn = fn / np.sum(fn**2, axis=1)[:,None]**.5
		return fn


	def subdivide(self): # by midpoint of every edge to maintain good triangulation
		V,F = self.vertices, self.faces
		nv, nf = len(V), len(F)
		V0, V1, V2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
		V = np.vstack((V, V0/2+V1/2, V1/2+V2/2, V0/2+V2/2))
		ind01, ind12, ind02 = np.arange(nv,nv+nf), np.arange(nv+nf,nv+nf*2), np.arange(nv+nf*2,nv+nf*3)
		f0 = np.vstack((ind02,F[:,0],ind01))
		f1 = np.vstack((ind01,F[:,1],ind12))
		f2 = np.vstack((ind12,F[:,2],ind02))
		fc = np.vstack((ind02,ind01,ind12))
		F = np.hstack((f0,f1,f2,fc)).T
		return self.__class__(V,F)


	@classmethod
	def read(cls, file, **initargs):
		if file.endswith('.stl'):
			with open(file, 'rb') as f:
				f.seek(80)
				data = f.read()
			nf, data = struct.unpack('I', data[0:4])[0], data[4:]
			data = struct.unpack('f'*(nf*12), b''.join([data[i*50:i*50+48] for i in range(nf)]))
			data = np.asarray(data).reshape(-1,12)
			FN = data[:,0:3]
			V = data[:,3:12].reshape(-1,3)
			F = np.arange(0,len(V)).reshape(-1,3)
			s = cls(V, F, **initargs)
		else:
			s = cls.read_npz(file)
		return s


	@classmethod
	def from_quadrilateral(cls, vertices, faces, **initargs):
		pass


	@classmethod
	def from_vtk(cls, polydata):
		V = PointArray(polydata.GetPoints())
		F = np.frombuffer(
				polydata.GetPolys().GetData(), 
				dtype=np.int64
			).reshape(-1,4)[:,1:]
		return cls(V,F)


	def to_vtk(self):
		polydata = vtkPolyData()
		points = vtkPoints()
		points.SetData(numpy_to_vtk(self.vertices, array_type=vtk.VTK_DOUBLE))
		polydata.SetPoints(points)
		polys = vtkCellArray()
		polys.SetData(3,numpy_to_vtk(self.faces.ravel()))
		polydata.SetPolys(polys)
		return polydata


	def write(self, file):
		if file.endswith('.stl'):
			data = np.hstack((self.face_normals, self.vertices[self.faces[:,0]], self.vertices[self.faces[:,1]], self.vertices[self.faces[:,2]])).tolist() # to write in single precision
			bs = bytearray(80)
			bs += struct.pack('I', len(data))
			bs += b''.join( [struct.pack('f'*len(d), *d) + b'\x00\x00' for d in data] )
			with open(file, 'wb') as f:
				f.write(bs)
		else:
			super().write(file)


	def actor(self, color=None, **kwargs):
		polyd = self.to_vtk()
		mapper = vtkPolyDataMapper()
		mapper.SetInputData(polyd)
		actor = vtkActor()
		actor.SetMapper(mapper)
		if color is None:
			color = (0.98, 0.50, 0.45)
		actor.GetProperty().SetColor(color)
		return actor


	

def test():
	srfc = TriangleSurface(
			PointArray(((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
			((0,1,2),(0,2,3))
		)
	print('vertices: \n', srfc.vertices)
	print('faces: \n', srfc.faces)

	srfc_new = TriangleSurface.from_vtk(srfc.to_vtk())
	if not np.allclose(srfc.vertices,srfc_new.vertices):
		print('vertices problem')
	if not np.allclose(srfc.faces,srfc_new.faces):
		print('faces problem')

	from ..visualization import Window
	w = Window()
	brst = TriangleSurface.read(r'C:\data\midsagittal\skin_smooth_10mm_cut.stl')

	w.renderer.AddActor(brst.actor())
	w.start()


if __name__=='__main__':
	pass