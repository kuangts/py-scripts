from vtkmodules.vtkCommonCore import vtkFloatArray, vtkDoubleArray, vtkAbstractArray,  vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.util.vtkConstants import *
from vtkmodules.util.numpy_support import *


'''`numpy_buffer` is used to stress that such numpy objects share memory with vtk, although functions in numpy_support do as well
different from deep copied numpy object with no outside references
buffer bridges to and from vtk
remember to only use basic indexing when assigning the result back to numpy array
remember to avoid size change
generally speaking, do not use these functions to track vtk object throughout
assign back result and set Modified() as soon as computation is finished'''



def numpy_to_vtk(num_array:numpy.ndarray, number_of_tuples=None, number_of_components=None):
    """Converts a real numpy Array to a VTK array object, rewritten by kuang 04/13

    This function only works for real arrays.
    Complex arrays are NOT handled.  It also works for multi-component
    arrays.  However, only 1, and 2 dimensional arrays are supported.
    This function is very efficient, so large arrays should not be a
    problem.

    WARNING: You must maintain a reference to the passed numpy array,
    if the numpy data is gc'd and VTK will point to garbage which will
    in the best case give you a segfault.

    Parameters:

    num_array
      a 1D or 2D, real numpy array.

    """

    z = num_array
    if not z.flags.c_contiguous:
        raise ValueError('Only contiguous arrays are supported.')
    
    if len(z.shape) >= 3 or len(z.shape) == 0:
        raise ValueError("Only 1-d and 2-d arrays are allowed.")
    
    if numpy.issubdtype(z.dtype, numpy.dtype(complex).type):
        raise ValueError("Complex arrays are not allowed.")
    
    # First create an array of the right type.
    result_array = create_vtk_array(get_vtk_array_type(z.dtype))

    # Find the shape and set number of components.
    if number_of_tuples is None:
        number_of_tuples = z.shape[0]
    if number_of_components is None:
        number_of_components = z.shape[1] if len(z.shape) > 1 else 1
    result_array.SetNumberOfTuples(number_of_tuples)
    result_array.SetNumberOfComponents(number_of_components)

    # Point the VTK array to the numpy data.  The last argument (1)
    result_array.SetVoidArray(z.ravel(), z.size, 1)
    result_array._numpy_reference = z
        
    return result_array


def numpy_to_vtkIdTypeArray(num_array:numpy.ndarray):
    isize = vtkIdTypeArray().GetDataTypeSize()
    dtype = num_array.dtype
    if isize == 4:
        if dtype != numpy.int32:
            raise ValueError(
             'Expecting a numpy.int32 array, got %s instead.' % (str(dtype)))
    elif isize == 8:
        if dtype != numpy.int64:
            raise ValueError(
             'Expecting a numpy.int64 array, got %s instead.' % (str(dtype)))
    else:
        raise ValueError('cannot handle %s id type' % (str(isize)))

    return numpy_to_vtk(num_array, number_of_tuples=num_array.size, number_of_components=1)


def vtk_to_numpy(vtk_array):
    """Converts a VTK data array to a numpy array.

    Given a subclass of vtkDataArray, this function returns an
    appropriate numpy array containing the same data -- it actually
    points to the same data.

    WARNING: This does not work for bit arrays.

    Parameters

    vtk_array
      The VTK data array to be converted.

    """
    typ = vtk_array.GetDataType()

    if typ == vtkConstants.VTK_BIT:
        raise ValueError('Bit arrays are not supported.')
    else:
        try:
            dtype = get_numpy_array_type(typ)
        except:
            raise ValueError("Unsupported array type %s"%typ)
        
    num_array = numpy.frombuffer(vtk_array, dtype=dtype)
    num_array = num_array.reshape(
        vtk_array.GetNumberOfTuples(),
        vtk_array.GetNumberOfComponents()
        ).squeeze()
    
    assert num_array.__array_interface__['data'][0] == num_array.__array_interface__['data'][0], \
        ValueError('cannot convert to numpy array without copying')
    
    return num_array


def vtkpoints_to_numpy(pts:vtkPoints): # only for convenience
    return vtk_to_numpy(pts.GetData()) # a view


def numpy_to_vtkpoints(arr:numpy.ndarray):
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(arr))
    return pts


def vtkpolys_to_numpy(cll:vtkCellArray):
    # because of vtkCellArray's GetData() limitation
    # https://vtk.org/doc/nightly/html/classvtkCellArray.html#a2841af7d1aae4c8db8544b2317dc712b
    # modifying the returned numpy array has no effect on input cell array

    n = cll.IsHomogeneous()
    if n<=0: raise ValueError('vtkCellArray is empty or heterogeneous')
    return vtk_to_numpy(cll.GetData()).reshape(-1, n+1)[:,1:]


def numpy_to_vtkpolys(arr):
    n_com = arr.shape[1] if len(arr.shape)>1 else 1
    vtk_arr = numpy_to_vtkIdTypeArray(arr)
    cll = vtkCellArray()
    cll.SetData(n_com, vtk_arr)
    return cll


def swig_to_numpy(swig_ptr, len, vtk_data_type):
    # https://discourse.slicer.org/t/centering-on-a-segment-from-a-script/6209/7
    vtk_arr = vtkAbstractArray.CreateArray(vtk_data_type)
    vtk_arr.SetVoidArray(swig_ptr, len, 1)
    return vtk_to_numpy(vtk_arr)


def vtkmatrix4x4_to_numpy(mtx:vtkMatrix4x4):
    arr = swig_to_numpy(mtx.GetData(), 16, VTK_DOUBLE)
    return arr.reshape(4,4)



# def shape_hetero_cells(cll:numpy.ndarray):
#     # from vtk serialized form (3,*,*,*,4,*,*,*,...,5,*,*,*) to python list of lists
#     cll_shaped = []
#     cll = cll.tolist() # data is copied
#     while cll:
#         cll_shaped.append(cll[1:cll[0]+1])
#         cll = cll[cll[0]+1:]
#     return cll_shaped


# def vtkpolydata_to_numpy_buffer(polyd:vtkPolyData, deep=0, ):
#     pts = vtkpoints_to_numpy_buffer(polyd.GetPoints())
#     ids = vtkpoints_to_numpy_buffer(polyd.GetPolys())
#     ids = shape_cells(ids) # setting this has no effect on polydata
#     return pts, ids


def test():
    import vtk
    import numpy as np
    sph = vtk.vtkSphereSource()
    sph.Update()
    s = sph.GetOutput()

    # vtk -> numpy
    pts = vtkpoints_to_numpy(s.GetPoints())
    pts[0,0] = 100
    print(np.all(vtkpoints_to_numpy(s.GetPoints())==pts))

    polys = vtkpolys_to_numpy(s.GetPolys())
    polys[0,:] = 20,20,20
    print(np.all(vtkpolys_to_numpy(s.GetPolys())==polys))

    # numpy -> vtk
    arr = numpy_to_vtkpoints(pts)
    arr.SetPoint(0,(100,200,300))
    print(np.all(vtkpoints_to_numpy(arr)==pts))

    arr = numpy_to_vtkpolys(polys)
    # changing numpy array changes the vtk cell array as well
    polys[0,:] = 20,20,20
    print(np.all(vtkpolys_to_numpy(arr)==polys))
    print(np.all(vtkpolys_to_numpy(s.GetPolys())==polys))


if __name__=='__main__':
    test()


