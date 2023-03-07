#! /usr/bin/env python

import struct, copy, re, plotly, typing
import numpy as np
from collections.abc import Sequence

class _Base: 
	
	def copy(self):
		return copy.deepcopy(self)

	@classmethod
	def merge(cls, *args, **initargs):
		# does not alter input args
		s = cls()
		conn, pnts = s.connectivity, s.points
		for arg in args:
			conn = np.vstack((conn, arg.connectivity+pnts.shape[0])) 
			pnts = np.vstack((pnts, arg.points))
		
		return cls(pnts, conn, **initargs)


	def __init__(self, points, connectivity, remove_duplicate=True):
		assert ( not len(points)) == ( not len(connectivity)), 'must specify points and connectivity both, or neither'
		self.points = np.array(points, dtype=float)
		self.connectivity = np.array(connectivity, dtype=int)
		if remove_duplicate:
			self.remove_duplicate()

	def remove_duplicate(self):
		if not self.points.size or not self.connectivity.size:
			return
		self.points = self.points.round(decimals=6)
		self.points, ind = np.unique(self.points, axis=0, return_inverse=True)
		self.connectivity = np.unique(ind[self.connectivity], axis=0)
		f_unique, ind = np.unique(self.connectivity, return_inverse=True)
		self.connectivity = ind.reshape(self.connectivity.shape)
		self.points = self.points[f_unique,:]

	@classmethod
	def read(cls, file, **initargs): # convenience method to keep consistent use of attr 'read', meant to be overriden
		return cls.read_npz(file, **initargs)

	@classmethod
	def read_npz(cls, file, **initargs): # meant to be inherited not overriden
		d = dict(np.load(file))
		return cls(d.pop('points'), d.pop('connectivity'), **d, **initargs)

	def write(self, file, **kwargs):
		np.savez(file, points=self.points, connectivity=self.connectivity, **kwargs)

	def __repr__(self):
		s = f'points: {self.points.shape}\n{self.points}\n' + f'connectivity: {self.connectivity.shape}\n{self.connectivity}\n'
		return s

	def __iter__(self):
		yield self.points
		yield self.connectivity

	def __len__(self):
		return 2


class TriangleSurface(_Base):

	def __init__(self, V=np.empty((0,3)), F=np.empty((0,3)), **initargs):
		F = np.array(F, dtype=int)
		if F.size and F.shape[1]==4:
			F = np.vstack((F[:,[0,1,2]],F[:,[0,2,3]]))
		super().__init__(V, F, **initargs)

	@property
	def V(self):
		return self.points

	@property
	def F(self):
		return self.connectivity

	@property
	def FN(self):
		v10 = self.V[self.F[:,2],:] - self.V[self.F[:,0],:]
		v20 = self.V[self.F[:,-1],:] - self.V[self.F[:,0],:]
		fn = np.cross(v10, v20)
		fn = fn / np.sum(fn**2, axis=1)[:,None]**.5
		return fn

	def subdivide(self, **initargs): # by midpoint of every edge to maintain good triangulation
		V,F = self.V, self.F
		nv, nf = len(V), len(F)
		V0, V1, V2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
		V = np.vstack((V, V0/2+V1/2, V1/2+V2/2, V0/2+V2/2))
		ind01, ind12, ind02 = np.arange(nv,nv+nf), np.arange(nv+nf,nv+nf*2), np.arange(nv+nf*2,nv+nf*3)
		f0 = np.vstack((ind02,F[:,0],ind01))
		f1 = np.vstack((ind01,F[:,1],ind12))
		f2 = np.vstack((ind12,F[:,2],ind02))
		fc = np.vstack((ind02,ind01,ind12))
		F = np.hstack((f0,f1,f2,fc)).T
		return self.__class__(V,F,remove_duplicate=False) # no duplicate

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
			s = cls(V=V, F=F, **initargs)
		else:
			s = cls.read_npz(file)
		return s

	def write(self, file):
		if file.endswith('.stl'):
			data = np.hstack((self.FN, self.V[self.F[:,0]], self.V[self.F[:,1]], self.V[self.F[:,2]])).tolist() # to write in single precision
			bs = bytearray(80)
			bs += struct.pack('I', len(data))
			bs += b''.join( [struct.pack('f'*len(d), *d) + b'\x00\x00' for d in data] )
			with open(file, 'wb') as f:
				f.write(bs)
		else:
			super().write(file)

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
		return trc

class HexahedralMesh(_Base):

	def __init__(self, N=np.empty((0,3)), E=np.empty((0,8)), NG=None, grid_seed=None, **initargs):
		super().__init__(N, E, **initargs)
		if NG is not None and len(NG):
			self.NG = NG
		elif isinstance(NG, typing.Callable):
			self.NG = NG(N,E)
		else:
			self.calculate_node_grid(seed=grid_seed)

	@property
	def N(self):
		return self.points

	@property
	def E(self):
		return self.connectivity
	
	@property
	def NG(self):
		if not hasattr(self, '_NG') or not np.array_equal(self._NG.shape, self.N.shape):
			self.calculate_node_grid()
		return self._NG

	@NG.setter
	def NG(self, _NG):
		assert np.array_equal(_NG.shape, self.N.shape)
		setattr(self, '_NG', _NG)

	@property
	def EG(self):
		eg = self.NG[self.E[:],:].reshape(*self.E.shape,3).mean(axis=1)
		return np.round(eg - np.min(eg, axis=0)).astype(int)

	@property
	def G3D(self):
		if not hasattr(self, '_G3D') or not len(self._G3D):
			self._G3D = -np.ones(self.NG.max(axis=0)+1, dtype=int)
			self._G3D[(*self.NG.T,)] = np.arange(self.NG.shape[0])
		return self._G3D

	def faces(self, node_index=None, element_index=None):
		if not self.E.size or not self.N.size:
			return np.empty((0,4))
		g = self.NG[self.E[0,:],:]
		g = g-g.mean(axis=0)
		column_index = [d.nonzero()[0] for d in (g<0).T] + [d.nonzero()[0] for d in (g>0).T]
		for i,c in enumerate(column_index):
			gc = np.delete(g[c], i%3, axis=1)
			c = c[np.argsort(np.sign(gc[:,0])+(gc[:,0] != gc[:,1]))]
			column_index[i] = c
			sign = np.dot(np.cross( g[c[1]]-g[c[0]], g[c[-1]]-g[c[0]], axis=0), g[c].mean(axis=0)) > 0
			if not sign:
				column_index[i] = c[::-1]
		E = self.E if element_index is None else self.E[element_index,:]
		f = np.vstack([ E[:,ind] for ind in column_index ])
		f = np.unique([np.roll(ff, -np.argmin(ff)) for ff in f], axis=0) # find unique faces up to a cyclic permutation
		if node_index is not None:
			if np.all(np.isin(node_index, [True,False])):
				node_index = node_index.nonzero()[0]
			f = f[np.all(np.isin(f, node_index), axis=1),:]
		return f

	def calculate_node_grid(self, seed=None): # alters self, to be subclassed
		ng = np.empty(self.N.shape)
		ng[:] = np.nan
		self.NG = ng

		if seed is None: # no means no
			return
		elif isinstance(seed, Sequence):
			if len(seed):
				pass
			elif hasattr(self, 'seed_grid'):
				seed = self.seed_grid()
			else:
				return
		elif isinstance(seed, typing.Callable):
			seed = seed(self.N, self.E)
		else:
			return

		seed = np.array(seed, dtype=float)
		seed -= np.min(seed, axis=0)
		seed /= (np.max(seed, axis=0)-np.min(seed, axis=0))
		seed -= np.mean(seed, axis=0)
		# calculate grid by expending from seed
		newly_set = np.array([0], dtype=int)
		ng[newly_set,:] = 0
		ind_unset = np.any(np.isnan(ng), axis=1)
		while np.any(ind_unset):
			elem_set = np.isin(self.E, newly_set)
			row_num = np.any(elem_set, axis=1)
			elem, elem_set = self.E[row_num], elem_set[row_num]
			for row in range(elem_set.shape[0]):
				col = elem_set[row].nonzero()[0][0]
				ng[elem[row],:] = seed + (ng[elem[row,col],:] - seed[col,:])
			newly_set = np.intersect1d(elem, np.where(ind_unset)[0])
			ind_unset[newly_set] = False

		ng -= np.min(ng, axis=0)
		ng = ng.round().astype(int)
		self.NG = ng


class SoftTissueMesh(HexahedralMesh):

	# the two different element structure
	G1 = np.asarray([[1,1,1],[1,0,1],[1,0,0],[1,1,0],[0,1,1],[0,0,1],[0,0,0],[0,1,0]], dtype=int)
	G2 = np.asarray([[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=int)


	@classmethod
	def read(cls, file, **initargs):
		nodes, elems = [],[]
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
			nodes4 = [node.split(',') for node in nodes.strip().split('\n')]
			nodes = np.asarray(nodes4, dtype=float)[:,1:]
			elems9 = [elem.split(',') for elem in elems.strip().split('\n')]
			elems = np.asarray(elems9, dtype=int)[:,1:]-1

		elif file.endswith('.feb'):
			with open(file,'r') as f:
				Nodes, Elements = re.search(r"<Nodes[\S ]*>\s*(<.*>)\s*</Nodes>.*<Elements[\S ]*\"hex8\"[\S ]*>\s*(<.*>)\s*</Elements>", f.read(), flags=re.MULTILINE|re.DOTALL).groups()
				nodes = re.findall(r'<node id[= ]*"(\d+)">(' + ','.join([r" *[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)+ *"]*3) + r')</node>', Nodes, flags=re.MULTILINE|re.DOTALL)
				elems = re.findall(r'<elem id[= ]*"(\d+)">(' + ','.join([r' *\d+ *']*8) + r')</elem>', Elements, flags=re.MULTILINE|re.DOTALL)

			nodeid = np.asarray([int(nd[0]) for nd in nodes])-1
			nodes_with_id = np.asarray([nd[1].split(',') for nd in nodes], dtype=float)
			elems = np.asarray([el[1].split(',') for el in elems], dtype=int)-1
			nodes = np.empty((nodeid.max()+1,3))
			nodes[:] = np.nan
			nodes[nodeid,:] = nodes_with_id[:]
	
		elif file.endswith('.npz'):
			return cls.read_npz(file, **initargs)

		s = cls(nodes, elems, **initargs)
		return s


	def seed_grid(self):
		N, E = self.N, self.E
		ng = np.zeros_like(N)
		# assumes u,v,w correspond to x,y,z, in both dimension and direction
		# first, find 8 nodes with single occurence
		occr = np.asarray([0]*N.shape[0])
		for x in N.flat:
			occr[x] += 1
		n8 = np.where(occr==1)[0]
		assert n8.size==8, 'check mesh'
		# then, set the grid position of these eight nodes
		# set left and right (-x -> -INF, +x -> +INF)
		ng[n8,0] = np.where(N[n8,0]<np.median(N[n8,0]), np.NINF, np.PINF)
		# set up and down (-z -> -INF, +z -> +INF)
		ng[n8,2] = np.where(N[n8,2]<np.median(N[n8,2]), np.NINF, np.PINF)
		# set front and back
		n4 = n8[N[n8,2]<np.median(N[n8,2])] # top 4
		c = N[n4].mean(axis=0)
		n2 = n4[N[n4,0]<np.median(N[n4,0])] # top left
		d = np.sum((N[n2] - c)**2, axis=1)**.5
		ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
		n2 = n4[N[n4,0]>np.median(N[n4,0])] # top right
		d = np.sum((N[n2] - c)**2, axis=1)**.5
		ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

		n4 = n8[N[n8,2]>np.median(N[n8,2])] # bottom 4
		c = N[n4].mean(axis=0)
		n2 = n4[N[n4,0]<np.median(N[n4,0])] # bottom left
		d = np.sum((N[n2] - c)**2, axis=1)**.5
		ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)
		n2 = n4[N[n4,0]>np.median(N[n4,0])] # bottom right
		d = np.sum((N[n2] - c)**2, axis=1)**.5
		ng[n2,1] = np.where(d<d.mean(), np.NINF, np.PINF)

		seed = np.ones((8,3))*.5
		ind_preset = np.all(np.isinf(ng), axis=1).nonzero()[0]
		for row,col in zip(*np.where(np.isin(N, ind_preset))):
			seed[col] *= np.sign(ng[N[row,col]])
		return seed


if __name__=='__main__':
	x, sx = np.linspace(-1,1,10, retstep=True)
	y, sy = np.linspace(-1,0,20, retstep=True)
	z, sz = np.linspace(-1,3,30, retstep=True)
	c = np.asarray(np.meshgrid(x,y,z, indexing='ij')).ravel('F').reshape(-1,3)
	s = box(edge_length=(sx,sy,sz), center=c)
	g3d = s.G3D
	ind = np.ones_like(g3d, dtype=bool)
	ind[1:,1:,1:-1] = False
	TriangleSurface(s.N, s.faces(node_index=g3d[ind])).plot(flatshading=True)

	s = SoftTissueMesh.read(r".\test\test.inp")
	g3d = s.G3D
	ind = np.ones_like(g3d, dtype=bool)
	ind[1:-1,1:-1,1:-1] = False
	sf = TriangleSurface(s.N,s.faces(node_index=s.NG[:,1]==2, element_index=s.EG[:,1]==1))
	sf.plot()

