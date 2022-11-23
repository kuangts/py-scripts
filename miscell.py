

# def box(edge_length=(1,1,1), center=np.zeros((1,3))):
	
# 	center = np.array(center).reshape(-1,3)
# 	N = np.array(np.meshgrid((-1,1),(-1,1),(-1,1))).T.reshape(-1,3)*.5*edge_length
# 	ng = N
# 	N = N - N.mean(axis=0)
# 	E = np.asarray([0,1,2,3,4,5,6,7], dtype=int).reshape(1,8)
# 	cub = [HexahedralMesh(N+c, E, remove_duplicate=False) for c in center]
# 	cub = HexahedralMesh.merge(*cub, grid_seed=ng)
# 	return cub


