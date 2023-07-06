from tools.select import HexMeshSurfaceSelector
from tkinter import Tk, filedialog
Tk().withdraw()
from vtk_bridge import *
from tools.mesh import write_inp


if __name__ == '__main__':
    case_dir = r'C:\data\20230501\n0044'
    sel = HexMeshSurfaceSelector(rf'{case_dir}\hexmesh_open.inp')
    sel.start()

    with filedialog.asksaveasfile(
            title='write .inp ...', 
            initialdir=case_dir, 
            initialfile='hexmesh_open_smoothed.inp') as f:
        nodes = vtkpoints_to_numpy_(sel.nodes)
        elems = vtkpolys_to_numpy_(sel.elems)
        write_inp(f, nodes, elems)
