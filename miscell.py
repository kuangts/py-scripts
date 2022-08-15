from mesh import TriangleSurface, HexahedralMesh
import numpy as np
import shutil, os

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

	return TriangleSurface(V, F)


def sphere(radius=1., center=np.zeros((1,3)), max_edge_length=None):
	# subdivide a regular icosahedron iteratively until
	# edge length < max_edge_length * radius
	center = np.array(center).reshape(-1,3)
	max_edge_length = .3 if max_edge_length is None else max_edge_length/radius
	s = regular_icosahedron()
	while np.sum((s.V[s.F[0,0]] - s.V[s.F[0,1]])**2)**.5 > max_edge_length:
		s = s.subdivide()
		s.V[:] = s.V / np.sum(s.V**2, axis=1)[:,None]**.5
	sph = [TriangleSurface(s.V*radius+c, s.F, remove_duplicate=False) for c in center]
	sph = TriangleSurface.merge(*sph)
	return sph


def box(edge_length=(1,1,1), center=np.zeros((1,3))):
	
	center = np.array(center).reshape(-1,3)
	N = np.array(np.meshgrid((-1,1),(-1,1),(-1,1))).T.reshape(-1,3)*.5*edge_length
	ng = N
	N = N - N.mean(axis=0)
	E = np.asarray([0,1,2,3,4,5,6,7], dtype=int).reshape(1,8)
	cub = [HexahedralMesh(N+c, E, remove_duplicate=False) for c in center]
	cub = HexahedralMesh.merge(*cub, grid_seed=ng)
	return cub


def find_files(list_of_necessary_keywords, in_list_of_files, copy_to_dir=''):
    files = in_list_of_files
    for wd in list_of_necessary_keywords:
        files = [f for f in files if wd.strip() in f]
    if copy_to_dir:
        try:
            os.makedirs(copy_to_dir, exist_ok=True)
        except:
            print('check ' + copy_to_dir)
        else:
            for file in files:
                shutil.copy(file, copy_to_dir)
    print('found', len(files), 'for', *list_of_necessary_keywords)
    return files

