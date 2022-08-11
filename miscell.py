from .mesh import TriangleSurface, HexahedralMesh
import numpy as np
import shutil, os


Labels = {
		'Detached':[
			'S', 'U0R', 'U1R-R', 'U1R-L', 'L0R', 'L1R-R', 'L1R-L', 'COR-R', 'COR-L'
			],
		'Cranium':[
			'Gb', 'N', 'OrM-R', 'OrM-L', 'SOF-R', 'SOF-L', 'Rh'
			],
		'ZMC Right':[
			'SOr-R', 'OrS-R', 'Ft-R', 'Fz-R', 'Or-R', 'ION-R', 'J-R', 'Zy-R', 'Po-R'
			],
		'ZMC Left':[
			'SOr-L', 'OrS-L', 'Ft-L', 'Fz-L', 'Or-L', 'ION-L', 'J-L', 'Zy-L', 'Po-L'
			],
		'Maxilla':[
			'ANS', 'A', 'IC', 'GPF-R', 'GPF-L', 'PNS'
			],
		'Ramus Right':[
			'Cr-R', 'SIG-R', 'Co-R', 'COR-R', 'RMA-R', 'RP-R', 'Gos-R', 'Go-R', 'Goi-R', 'Ag-R'
			],
		'Ramus Left':[
			'Cr-L', 'SIG-L', 'Co-L', 'COR-L', 'RMA-L', 'RP-L', 'Gos-L', 'Go-L', 'Goi-L', 'Ag-L'
			],
		'Mandible':[
			'B', 'Pg', 'Gn', 'Me', 'MF-R', 'MF-L'
			],
		'Upper Teeth':[
			'U1R-R', 'U1R-L', 'U0R', 'U0',
			'U1E-R', 'U1E-L',
			'U2E-R', 'U2E-L',
			'U3T-R', 'U3T-L',
			'U4BC-R', 'U4BC-L', 'U4LC-R', 'U4LC-L',
			'U5BC-R', 'U5BC-L', 'U5LC-R', 'U5LC-L',
			'U6CF-R', 'U6CF-L', 'U6MBC-R', 'U6MBC-L', 'U6MLC-R', 'U6MLC-L', 'U6DBC-R', 'U6DBC-L', 'U6DLC-R', 'U6DLC-L',
			'U7MBC-R', 'U7MBC-L', 'U7MLC-R', 'U7MLC-L', 'U7DBC-R', 'U7DBC-L', 'U7DLC-R', 'U7DLC-L'
			],
		'Lower Teeth':[
			"L1R-R", "L1R-L", "L0R", "L0",
			'L1E-R', 'L1E-L',
			'L2E-R', 'L2E-L',
			'L3T-R', 'L3T-L',
			'L34Embr-R', 'L34Embr-L',
			'L4BC-R', 'L4BC-L', 'L4LC-R', 'L4LC-L',
			'L5BC-R', 'L5BC-L', 'L5LC-R', 'L5LC-L',
			'L6CF-R', 'L6CF-L', 'L6MBC-R', 'L6MBC-L', 'L6MLC-R', 'L6MLC-L', 'L6DBC-R', 'L6DBC-L', 'L6DLC-R', 'L6DLC-L', 'L6DC-R', 'L6DC-L',
			'L7MBC-R', 'L7MBC-L', 'L7MLC-R', 'L7MLC-L', 'L7DBC-R', 'L7DBC-L', 'L7DLC-R', 'L7DLC-L'
			],
		'Cranial Base':[
			'S', 'Ba', 'FMP', 'M-R', 'M-L', 'SMF-R', 'SMF-L', 'GFC-R', 'GFC-L'
			],
		'Soft Tissue':[
			"Gb'", "N'", "En-R", "En-L", "Ex-R", "Ex-L", "Mf-R", "Mf-L", "OR'-R", "OR'-L",
			"Prn", "AL-R", "AL-L", "Ac-R", "Ac-L", "Zy'-R", "Zy'-L", "CM", "Nt-R", "Nt-L",
			"Nb-R", "Nb-L", "Sn", "A'", "Ss", "Ls", "Cph-R", "Cph-L", "Stm-U", "Stm-L",
			"Ch-R", "Ch-L", "Li", "Sl", "B'", "Pog'", "Gn'", "Me'", "C", "Go'-R", "Go'-L"
			],
		'Soft Tissue--PostOp':[
			"Gb'", "N'", "En-R", "En-L", "Ex-R", "Ex-L", "Mf-R", "Mf-L", "OR'-R", "OR'-L",
			"Prn--PostOp", "AL-R--PostOp", "AL-L--PostOp", "Ac-R--PostOp", "Ac-L--PostOp", "Zy'-R--PostOp", "Zy'-L--PostOp", "CM--PostOp", "Nt-R--PostOp", "Nt-L--PostOp",
			"Nb-R--PostOp", "Nb-L--PostOp", "Sn--PostOp", "A'--PostOp", "Ss--PostOp", "Ls--PostOp", "Cph-R--PostOp", "Cph-L--PostOp", "Stm-U--PostOp", "Stm-L--PostOp",
			"Ch-R--PostOp", "Ch-L--PostOp", "Li--PostOp", "Sl--PostOp", "B'--PostOp", "Pog'--PostOp", "Gn'--PostOp", "Me'--PostOp", "C--PostOp", "Go'-R--PostOp", "Go'-L--PostOp"
			]
	}


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

