from vtkmodules.vtkCommonCore import vtkAbstractArray, vtkIdTypeArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.util.vtkConstants import *
from vtkmodules.util.numpy_support import *
import sys

'''convenience methods to quickly set, get, nmodify data between numpy and vtk
"_*" method output always shares memory with input; the rest never do
make sure there is no namespace conflict with vtkmodules.util.numpy_support
set Modified() on vtk objects afterwards to update pipeline
'''


VTK_ID_DTYPE = ID_TYPE_CODE

def numpy_to_vtk_(num_array, number_of_tuples=None, number_of_components=None):
    
    # validate input numpy array
    # type
    if numpy.issubdtype(num_array.dtype, numpy.dtype(complex).type):
        raise ValueError("Only real arrays are not allowed.")
    
    # shape
    if len(num_array.shape) >= 3 or len(num_array.shape) == 0:
        raise ValueError("Only 1-d and 2-d arrays are allowed.")
    if number_of_tuples is None:
        number_of_tuples = num_array.shape[0]
    if number_of_components is None:
        number_of_components = num_array.shape[1] if len(num_array.shape) > 1 else 1

    # continuity
    if not num_array.flags.c_contiguous:
        raise ValueError('Only c-contiguous arrays are supported. Call numpy.ascontinuous first.')
    
    # ravel numpy array
    num_array_flat = num_array.ravel()
    if num_array_flat.__array_interface__['data'][0] != num_array.__array_interface__['data'][0]:
        raise ValueError('Cannot ravel numpy array without copying. Try calling numpy.ascontiguous and make it c-contiguous first.')
        
    # Create an vtk array of the right type.
    result_array = create_vtk_array(get_vtk_array_type(num_array.dtype))
    result_array.SetNumberOfTuples(number_of_tuples)
    result_array.SetNumberOfComponents(number_of_components)
    # save=1 to prevent system from reallocating memory since numpy array owns the data
    result_array.SetVoidArray(num_array_flat, number_of_components*number_of_tuples, 1)
    # numpy array should live as long as vtk array lives, so we maintain a ref to numpy array
    result_array._numpy_reference = num_array

    return result_array


def numpy_to_vtkIdTypeArray_(num_array:numpy.ndarray, number_of_tuples=None, number_of_components=None):
    """shares momery of an int numpy array with a VTK index array object

    Parameters:

    num_array
      a 1D or 2D, numpy.int32/numpy.int64 (depending on the system) numpy array.

    """

    if num_array.dtype != VTK_ID_DTYPE:
        raise ValueError(f'Expecting a {VTK_ID_DTYPE} array, got {num_array.dtype} instead.')
            
    return numpy_to_vtk_(num_array, number_of_tuples, number_of_components)


def vtk_to_numpy_(vtk_array, number_of_tuples=None, number_of_components=None):
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
        
    if number_of_tuples is None:
        number_of_tuples = vtk_array.GetNumberOfTuples()
    if number_of_components is None:
        number_of_components = vtk_array.GetNumberOfComponents()

    # create numpy array from vtk array memory
    num_array = numpy.frombuffer(vtk_array, dtype=dtype)
    num_array_shaped = num_array.reshape(number_of_tuples, number_of_components)
    
    # make sure output numpy array and input vtk array point to same data
    if num_array_shaped.__array_interface__['data'][0] != num_array.__array_interface__['data'][0]:
        raise ValueError('cannot convert to numpy array without copying')
    
    return num_array_shaped


def vtkpoints_to_numpy_(pts:vtkPoints):
    """wraps vtk_to_numpy_() for convenience
    
    """
    return vtk_to_numpy_(pts.GetData())


def numpy_to_vtkpoints_(arr:numpy.ndarray, vtk_pts:vtkPoints=None):
    """wraps numpy_to_vtk_() for convenience
    
    for ease of use:
        if vtk_pts is given, perform in-place modification and return None
        else create new vtkPoints instance and return it

    """
    pts = vtkPoints() if vtk_pts is None else vtk_pts
    pts.SetData(numpy_to_vtk_(arr))
    return pts if vtk_pts is None else pts.Modified() 


def vtkpolys_to_numpy_(cll:vtkCellArray): 
    """wraps vtk_to_numpy_() for convenience
    avoid size change on numpy or vtk object

    """

    n = cll.IsHomogeneous()
    if n<=0: 
        raise ValueError('vtkCellArray is empty or heterogeneous')
    
    return vtk_to_numpy_(cll.GetConnectivityArray(), -1, n)


def numpy_to_vtkpolys_(arr:numpy.ndarray, vtk_cll:vtkCellArray=None):
    """wraps numpy_to_vtkIdTypeArray_() for convenience
    avoid size change on numpy or vtk object
    
    """
    if vtk_cll is None:
        cll = vtkCellArray()
        number_of_components = arr.shape[1] if len(arr.shape)>1 else 1
        offsets = cll.GetOffsetsArray().NewInstance()
        offsets.DeepCopy(numpy_to_vtkIdTypeArray_(numpy.arange(0, arr.size+1, number_of_components, dtype=VTK_ID_DTYPE)))
        cll.SetData(offsets, numpy_to_vtkIdTypeArray_(arr, arr.size, 1))
        return cll
    else:
        vtk_cll.SetData(vtk_cll.GetOffsetsArray(), numpy_to_vtkIdTypeArray_(arr, arr.size, 1))


def swig_to_numpy_(swig_ptr, len, vtk_data_type):
    """converts a swig pointer (e.g. _000001bfd4897340_p_void) to a numpy array
    
    """
    # https://discourse.slicer.org/t/centering-on-a-segment-from-a-script/6209/7
    vtk_arr = vtkAbstractArray.CreateArray(vtk_data_type)
    vtk_arr.SetVoidArray(swig_ptr, len, 1)
    return vtk_to_numpy_(vtk_arr)


def vtkmatrix4x4_to_numpy_(mtx:vtkMatrix4x4):
    """shares a 4x4 vtk matrix with a numpy array
    wraps numpy_to_swig_()
    
    """
    return swig_to_numpy_(mtx.GetData(), 16, VTK_DOUBLE).reshape(4,4)



def test():
    from vtk import vtkSphereSource, vtkTriangle 

    sph = vtkSphereSource()
    sph.Update()
    s = sph.GetOutput()

    # points
    pts = vtkpoints_to_numpy_(s.GetPoints())
    pts[0,:] += 1
    s.GetPoints().SetPoint(1,(0,0,0))
    pts = pts.copy()
    numpy_to_vtkpoints_(pts, s.GetPoints())
    pts[2,:] += 1
    s.GetPoints().SetPoint(3,(0,0,0))
    arr = numpy_to_vtkpoints_(pts)
    arr.SetPoint(4,(0,0,0))
    if numpy.all(vtkpoints_to_numpy_(s.GetPoints())==pts):
        print('points checked')
    else:
        print('points wrong')


    # polys
    polys = vtkpolys_to_numpy_(s.GetPolys())
    polys[0,:] = 1,2,3
    s.GetPolys().ReverseCellAtId(1)
    polys = polys.copy()
    numpy_to_vtkpolys_(polys, s.GetPolys())
    polys[2,:] = 1,2,3
    s.GetPolys().ReverseCellAtId(3)
    arr = numpy_to_vtkpolys_(polys)
    arr.ReverseCellAtId(4)
    if numpy.all(vtkpolys_to_numpy_(s.GetPolys())==polys):
        print('polys checked')
    else:
        print('polys wrong')


if __name__=='__main__':
    test()


