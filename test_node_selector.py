from tools.ui import PolygonalSurfaceNodeSelector
from tkinter import Tk, filedialog
Tk().withdraw()
from vtk_bridge import *
from tools.mesh import write_inp
from tools.polydata import *


if __name__ == '__main__':
    sel = PolygonalSurfaceNodeSelector()
    stl_pick = polydata_from_stl(r'C:\data\clipped_with_mesh\n0034\mesh_surf.stl')
    stl_show = polydata_from_stl(r'C:\data\clipped_with_mesh\n0034\clipper_large.stl')
    sel.initialize(stl_pick)
    sel.start()


