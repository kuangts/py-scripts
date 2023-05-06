from vtkmodules.vtkCommonCore import vtkAbstractArray, vtkIdTypeArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.util.vtkConstants import *
from vtkmodules.util.numpy_support import *


'''convenience methods to quickly set, get, nmodify data between numpy and vtk
"share_*" method output always shares memory with input; the rest never do
make sure no namespace conflict with vtkmodules.util.numpy_support
set Modified() on vtk objects afterwards to update pipeline
'''


def share_numpy_to_vtk(num_array:numpy.ndarray, number_of_tuples=None, number_of_components=None):
    """shares momery of a real numpy array with a VTK array object

    WARNING: You must maintain a reference to the passed numpy array,
    if the numpy data is gc'd and VTK will point to garbage which will
    in the best case give you a segfault.

    Parameters:

    num_array
      a 1D or 2D, real numpy array.

    """

    # validate input numpy array
    z = num_array
    if not z.flags.c_contiguous: 
        raise ValueError('Only c-contiguous arrays are supported.Call numpy.ascontinuous first.')
        # otherwise ravel() will return a copy
    if numpy.issubdtype(z.dtype, numpy.dtype(complex).type):
        raise ValueError("Complex arrays are not allowed.")

    if len(z.shape) >= 3 or len(z.shape) == 0:
        raise ValueError("Only 1-d and 2-d arrays are allowed.")
    
    # Find the shape and set number of components.
    if number_of_tuples is None:
        number_of_tuples = z.shape[0]
    if number_of_components is None:
        number_of_components = z.shape[1] if len(z.shape) > 1 else 1
    
    # ravel numpy array
    z_flat = z.ravel()
    assert z_flat.__array_interface__['data'][0] == z.__array_interface__['data'][0], \
        ValueError('cannot convert to numpy array without copying')

    # Create an vtk array of the right type.
    result_array = create_vtk_array(get_vtk_array_type(z.dtype))
    result_array.SetNumberOfTuples(number_of_tuples)
    result_array.SetNumberOfComponents(number_of_components)
    result_array.SetVoidArray(z_flat, number_of_components*number_of_tuples, 1)

    # make sure data owner stays
    result_array._numpy_reference = z
        
    return result_array


def get_int_type():
    isize = vtkIdTypeArray().GetDataTypeSize()
    if isize == 4:
        return numpy.int32
    elif isize == 8:
        return numpy.int64
    else:
        raise ValueError('vtk error')


def share_numpy_to_vtkIdTypeArray(num_array:numpy.ndarray):
    """shares momery of an int numpy array with a VTK index array object

    WARNING: You must maintain a reference to the passed numpy array,
    if the numpy data is gc'd and VTK will point to garbage which will
    in the best case give you a segfault.

    Parameters:

    num_array
      a 1D or 2D, np.int32/np.int64 (depending on the system) numpy array.

    """

    itype = get_int_type()
    if num_array.dtype != itype:
        raise ValueError(f'Expecting a {itype} array, got {num_array.dtype} instead.')
    
    return share_numpy_to_vtk(num_array, number_of_tuples=num_array.size, number_of_components=1)


def share_vtk_to_numpy(vtk_array):
    """shares momery of vtk array object with a numpy array

    Given a subclass of vtkDataArray, this function returns an
    appropriate numpy array containing the same data -- it actually
    points to the same data.

    WARNING: This does not work for bit arrays.

    Parameters

    vtk_array
      The VTK data array to be converted.

    """

    # validate input vtk array
    typ = vtk_array.GetDataType()

    if typ == VTK_BIT:
        raise ValueError('Bit arrays are not supported.')
    else:
        try:
            dtype = get_numpy_array_type(typ)
        except:
            raise ValueError("Unsupported array type %s"%typ)

    # create numpy array from vtk array memory
    num_array = numpy.frombuffer(vtk_array, dtype=dtype)
    num_array_shaped = num_array.reshape(
        vtk_array.GetNumberOfTuples(),
        vtk_array.GetNumberOfComponents()
        ).squeeze()
    
    # make sure output numpy array and input vtk array point to same data
    if num_array_shaped.__array_interface__['data'][0] == num_array.__array_interface__['data'][0]:
        return num_array_shaped
    else:
        raise ValueError('cannot convert to numpy array without copying')


def share_vtkpoints_to_numpy(pts:vtkPoints):
    """wraps share_vtk_to_numpy() for convenience
    
    """
    return share_vtk_to_numpy(pts.GetData())


def share_numpy_to_vtkpoints(arr:numpy.ndarray, vtk_pts:vtkPoints=None):
    """wraps share_numpy_to_vtk() for convenience
    
    for ease of use:
        if vtk_pts is given, perform in-place modification and return None
        else create new vtkPoints instance and return it

    """
    pts = vtkPoints() if vtk_pts is None else vtk_pts
    pts.SetData(share_numpy_to_vtk(arr))
    return pts if vtk_pts is None else pts.Modified()


def share_vtkpolys_to_numpy(cll:vtkCellArray): 
    """wraps share_vtk_to_numpy() for convenience
    avoid size change on numpy or vtk object

    """

    n = cll.IsHomogeneous()
    if n<=0: 
        raise ValueError('vtkCellArray is empty or heterogeneous')
    
    arr = share_vtk_to_numpy(cll.GetConnectivityArray())
    arr_shaped = arr.reshape(-1, n)

    # make sure output numpy array and input vtk array point to same data
    if arr_shaped.__array_interface__['data'][0] == arr.__array_interface__['data'][0]:
        return arr_shaped
    else:
        raise ValueError('cannot convert to numpy array without copying')


def share_numpy_to_vtkpolys(arr, vtk_cll:vtkCellArray=None):
    """wraps share_numpy_to_vtkIdTypeArray() for convenience
    avoid size change on numpy or vtk object
    
    """
    if vtk_cll is None:
        cll = vtkCellArray() 
        n_com = arr.shape[1] if len(arr.shape)>1 else 1
        offset = numpy_to_vtkIdTypeArray(numpy.arange(0, arr.size+1, n_com).astype(get_int_type()), deep=1)
        cll.SetData(offset, share_numpy_to_vtkIdTypeArray(arr))
        return cll
    else:
        vtk_cll.SetData(vtk_cll.GetOffsetsArray(), share_numpy_to_vtkIdTypeArray(arr))


def share_swig_to_numpy(swig_ptr, len, vtk_data_type):
    """converts a swig pointer (e.g. _000001bfd4897340_p_void) to a numpy array
    
    """
    # https://discourse.slicer.org/t/centering-on-a-segment-from-a-script/6209/7
    vtk_arr = vtkAbstractArray.CreateArray(vtk_data_type)
    vtk_arr.SetVoidArray(swig_ptr, len, 1)
    return share_vtk_to_numpy(vtk_arr)


def share_vtkmatrix4x4_to_numpy(mtx:vtkMatrix4x4):
    """shares a 4x4 vtk matrix with a numpy array
    wraps share_swig_to_numpy()
    
    """
    return share_swig_to_numpy(mtx.GetData(), 16, VTK_DOUBLE).reshape(4,4)



def test():
    from vtk import vtkSphereSource, vtkTriangle

    sph = vtkSphereSource()
    sph.Update()
    s = sph.GetOutput()

    # points
    pts = share_vtkpoints_to_numpy(s.GetPoints())
    pts[0,:] += 1
    s.GetPoints().SetPoint(1,(0,0,0))
    pts = pts.copy()
    share_numpy_to_vtkpoints(pts, s.GetPoints())
    pts[2,:] += 1
    s.GetPoints().SetPoint(3,(0,0,0))
    arr = share_numpy_to_vtkpoints(pts)
    arr.SetPoint(4,(0,0,0))
    if numpy.all(share_vtkpoints_to_numpy(s.GetPoints())==pts):
        print('points checked')
    else:
        print('points wrong')


    # polys
    polys = share_vtkpolys_to_numpy(s.GetPolys())
    polys[0,:] = 1,2,3
    s.GetPolys().ReverseCellAtId(1)
    polys = polys.copy()
    share_numpy_to_vtkpolys(polys, s.GetPolys())
    polys[2,:] = 1,2,3
    s.GetPolys().ReverseCellAtId(3)
    arr = share_numpy_to_vtkpolys(polys)
    arr.ReverseCellAtId(4)
    if numpy.all(share_vtkpolys_to_numpy(s.GetPolys())==polys):
        print('polys checked')
        c = vtkCellArray()
        c.DeepCopy(arr)
        arr.Append(c)
        if numpy.all(share_vtkpolys_to_numpy(s.GetPolys())==polys):
            print('vtk polys size change might be ok')
        else:
            print('must prevent vtk polys size change')
    else:
        print('polys wrong')


if __name__=='__main__':
    test()


