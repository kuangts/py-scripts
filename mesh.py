#! /usr/bin/env python

import dependencies_test # test if all dependency requirements are met
import struct, sys, os, copy, re, plotly
import numpy as np

class Base: 
	
	def deepcopy(self):
		return copy.deepcopy(self)

	def copy(self):
		return self.deepcopy()

	@classmethod
	def merge(cls, *args):
		# does not alter input args
		s = cls()
		for arg in args:
			s.connectivity = np.vstack((s.connectivity, arg.connectivity+len(s.points))) 
			s.points = np.vstack((s.points, arg.points))

		s.remove_duplicate_points()
		return s

	def __init__(self, points=[], connectivity=[]):
		# do the following when subclassing
		# self.points, self.connectivity = np.empty(shape=(0,3),dtype=float), np.empty(shape=(0,3),dtype=int)
		assert ( not len(points)) == ( not len(connectivity)), 'must specify points and connectivity both, or neither'
		self.points = np.asarray(points, dtype=float)
		self.connectivity = np.asarray(connectivity, dtype=int)
		self.remove_duplicate_points()

	def remove_duplicate_points(self):
		self.points, ind = np.unique(self.points, axis=0, return_inverse=True)
		self.connectivity[:], ind = np.unique(ind[self.connectivity].reshape(self.connectivity.shape), axis=0, return_index=True)

	def __repr__(self):
		s = f'points: {self.points.shape}\n{self.points}\n' + f'connectivity: {self.connectivity.shape}\n{self.connectivity}\n'
		return s

class TriangleSurface(Base):

	def __init__(self, V=np.empty((0,3)), F=np.empty((0,3)), **kwargs):
		V = np.asarray(V, dtype=float)
		F = np.asarray(F, dtype=float)
		if F.size and F.shape[1]==4:
			F = np.vstack((F[:,[1,2,3]],F[:,[1,3,4]]))
		super().__init__(V, F)

	@property
	def V(self):
		return self.points

	@property
	def F(self):
		return self.connectivity

	@property
	def FN(self):
		v10 = self.V[self.F[:,1],:] - self.V[self.F[:,0],:]
		v20 = self.V[self.F[:,2],:] - self.V[self.F[:,0],:]
		fn = np.cross(v10, v20)
		fn = fn / np.sum(fn**2, axis=1)[:,None]**.5
		return fn

	def subdivide(self): # by midpoint of every edge so as to maintain good triangulation
		nv, nf = len(self.V), len(self.F)
		V0, V1, V2 = self.V[self.F[:,0]], self.V[self.F[:,1]], self.V[self.F[:,2]]
		V = self.V
		self.V = np.vstack((V, V0/2+V1/2, V1/2+V2/2, V0/2+V2/2))
		ind01, ind12, ind02 = np.arange(nv,nv+nf), np.arange(nv+nf,nv+nf*2), np.arange(nv+nf*2,nv+nf*3)
		f0 = np.vstack((ind02,self.F[:,0],ind01))
		f1 = np.vstack((ind01,self.F[:,1],ind12))
		f2 = np.vstack((ind12,self.F[:,2],ind02))
		fc = np.vstack((ind02,ind01,ind12))
		self.F = np.hstack((f0,f1,f2,fc)).T
		self.remove_duplicate_points()

	@classmethod
	def read(cls, file):
		with open(file, 'rb') as f:
			f.seek(80)
			data = f.read()
		nf, data = struct.unpack('I', data[0:4])[0], data[4:]
		data = struct.unpack('f'*(nf*12), b''.join([data[i*50:i*50+48] for i in range(nf)]))
		data = np.asarray(data).reshape(-1,12)
		FN = data[:,0:3]
		V = data[:,3:12].reshape(-1,3)
		F = np.arange(0,len(V)).reshape(-1,3)
		s = cls(V=V, F=F)
		s.remove_duplicate_points()
		return s

	def write(self, file):
		data = np.hstack((self.FN, self.V[self.F[:,0]], self.V[self.F[:,1]], self.V[self.F[:,2]])).tolist() # to write in single precision
		bs = bytearray(80)
		bs += struct.pack('I', len(data))
		bs += b''.join( [struct.pack('f'*len(d), *d) + b'\x00\x00' for d in data] )
		with open(file, 'wb') as f:
			f.write(bs)

	def plot(self, figure=None, **kwargs):
		# if 'intensity' not in kwargs:
		# 	kwargs['intensity'] = self.V[:,0]
		# 	kwargs['colorscale'] = [(0, "red"), (1, "blue")]
		xyz = dict(zip(['x','y','z'], self.V.T))
		ijk = dict(zip(['i','j','k'], self.F.T))
		trc = plotly.graph_objects.Mesh3d(**xyz, **ijk, **kwargs)
		fig = plotly.graph_objects.Figure() if figure is None else figure
		fig.add_trace(trc)
		if figure is None:
			fig.show()




















class SoftTissueMesh(Base):

	def __init__(self, N=[], E=[], element_structure=[], **kwargs):
		super().__init__(N,E, **kwargs)
		if not len(element_structure):
			self.element_structure = self.__class__.G1
		else:
			self.element_structure = element_structure
	@property
	def N(self):
		return self.points

	@property
	def E(self):
		return self.connectivity
	
	@property
	def NG(self):
		if not hasattr(self, '_NG') or not len(self._NG):
			self._NG = np.empty(self.N.shape)
			self._NG[:] = np.nan
			self.set_grid_orientation()
			self.update_grid()
		return self._NG

	@property
	def G3D(self):
		if not hasattr(self, '_G3D') or not len(self._G3D):
			self._G3D = -np.ones(self.NG.max(axis=0)+1, dtype=int)
			ind = np.ravel_multi_index(self.NG.T, self._G3D.shape)
			self._G3D.flat[ind] = np.arange(self.NG.shape[0])
		return self._G3D

	@classmethod
	def read(cls, file, ele_str=[]):
		if file.endswith('.inp'):
			with open(file,'r') as f:
				match = re.search(
					r'\*.*NODE[\S ]*\s+(.*)\*.*END NODE.*\*.*ELEMENT[\S ]*\s+(.*)\*.*END ELEMENT', 
					f.read(), 
					re.MULTILINE | re.DOTALL)
				try:
					nodes, elems = match.group(1), match.group(2)
				except Exception as e:
					print(e)
					raise ValueError('the file cannot be read for nodes and elements')
			nodes = [node.split(',') for node in nodes.strip().split('\n')]
			nodes = np.asarray(nodes, dtype=float)[:,1:]
			elems = [elem.split(',') for elem in elems.strip().split('\n')]
			elems = np.asarray(elems, dtype=int)[:,1:]-1
		else:
			nodes, elems = [],[]

		return cls(nodes, elems)

	# the two different element structure
	G1 = np.asarray([[1,1,1],[1,0,1],[1,0,0],[1,1,0],[0,1,1],[0,0,1],[0,0,0],[0,1,0]], dtype=int)
	G2 = np.asarray([[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=int)

	@classmethod
	def new(cls, size=[2,3,4], element_structure=None):
		x,y,z = np.meshgrid(range(size[0]),range(size[1]),range(size[2]), indexing='ij')
		g = np.arange(np.prod(x.shape)).reshape(x.shape)
		e = -np.ones((np.prod(np.asarray(x.shape)-1),8))
		s = element_structure if element_structure else cls.G1
		for i in range(e.shape[1]):
			e[:,i] = g[s[i,0]:g.shape[0]-1+s[i,0], s[i,1]:g.shape[1]-1+s[i,1], s[i,2]:g.shape[2]-1+s[i,2]].flatten()
		m = cls(np.vstack((x.flat,y.flat,z.flat)).T, e, element_structure=s)
		return m

	def faces(self, boundary=False):
		face_index = np.asarray([[4,3,2,1],[5,6,7,8],[1,2,6,5],[3,4,8,7],[2,3,7,6],[4,1,5,8]])
		f = np.vstack([ self.E[:,ind] for ind in face_index ])
		return f

	def set_grid_orientation(self):
		# assumes u,v,w correspond to x,y,z, in both dimension and direction
		# first, find 8 nodes with single occurence
		occr = np.asarray([0]*self.N.shape[0])
		for x in self.E.flat:
			occr[x] += 1
		n8 = np.where(occr==1)[0]
		assert n8.size==8, 'check mesh'
		# then, set the grid position of these eight nodes
		# set left and right (-x -> -INF, +x -> +INF)
		self._NG[n8,0] = np.where(self.N[n8,0]<np.median(self.N[n8,0]), np.NINF, np.PINF)
		# set up and down (-z -> -INF, +z -> +INF)
		self._NG[n8,2] = np.where(self.N[n8,2]<np.median(self.N[n8,2]), np.NINF, np.PINF)
		# set front and back
		n4 = n8[self.N[n8,2]<np.median(self.N[n8,2])] # top 4
		c = self.N[n4].mean(axis=0)
		n2 = n4[self.N[n4,0]<np.median(self.N[n4,0])] # top left
		d = np.sum((self.N[n2] - c)**2, axis=1)**.5
		self._NG[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
		n2 = n4[self.N[n4,0]>np.median(self.N[n4,0])] # top right
		d = np.sum((self.N[n2] - c)**2, axis=1)**.5
		self._NG[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

		n4 = n8[self.N[n8,2]>np.median(self.N[n8,2])] # bottom 4
		c = self.N[n4].mean(axis=0)
		n2 = n4[self.N[n4,0]<np.median(self.N[n4,0])] # bottom left
		d = np.sum((self.N[n2] - c)**2, axis=1)**.5
		self._NG[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
		n2 = n4[self.N[n4,0]>np.median(self.N[n4,0])] # bottom right
		d = np.sum((self.N[n2] - c)**2, axis=1)**.5
		self._NG[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

	def update_grid(self):
		ele_str = np.ones((8,3))*.5
		ind_preset = np.all(np.isinf(self._NG), axis=1).nonzero()[0]
		for row,col in zip(*np.where(np.isin(self.E, ind_preset))):
			ele_str[col] *= np.sign(self._NG[self.E[row,col]])
		self._NG[:] = np.nan
		newly_set = np.array([0], dtype=int)
		self._NG[newly_set,:] = 0
		ind_unset = np.any(np.isnan(self._NG), axis=1)
		while np.any(ind_unset):
			elem_set = np.isin(self.E, newly_set)
			row_num = np.any(elem_set, axis=1)
			elem, elem_set = self.E[row_num], elem_set[row_num]
			for row in range(elem_set.shape[0]):
				col = elem_set[row].nonzero()[0][0]
				self._NG[elem[row],:] = ele_str + (self._NG[elem[row,col],:] - ele_str[col,:])
			newly_set = np.intersect1d(elem, np.where(ind_unset)[0])
			ind_unset[newly_set] = False

		self._NG -= np.min(self._NG, axis=0)
		self._NG = np.round(self._NG).astype(int)


def sphere(radius=1., center=(0., 0., 0.), max_edge_length=.3):
	# subdivide a regular icosahedron iteratively until
	# edge length < max_edge_length * radius
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
				(-phi, 0, 1)])
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

		return TriangleSurface(V, F)

	s = regular_icosahedron()
	while np.sum((s.V[s.F[0,0]] - s.V[s.F[0,1]])**2)**.5 > radius * max_edge_length:
		s.subdivide()
		s.V = s.V / np.sum(s.V**2, axis=1)[:,None]**.5
	s.V = s.V * radius + center
	s.remove_duplicate_points()
	return s


