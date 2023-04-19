import struct
from collections import namedtuple
import numpy as np

stl = namedtuple('stl',('v','f','fn'))

def read_stl(file):

    with open(file, 'br') as f:
        f.seek(80)
        data = f.read()

    nf, data = struct.unpack('I', data[0:4])[0], data[4:]
    data = struct.unpack('f'*(nf*12), b''.join([data[i*50:i*50+48] for i in range(nf)]))
    data = np.asarray(data).reshape(-1,12)

    FN = data[:,0:3].astype(np.float32)
    V = data[:,3:12].reshape(-1,3).astype(np.float32)
    F = np.arange(0,len(V)).reshape(-1,3).astype(np.int64)

    return stl(v=V, f=F, fn=FN)


def write_stl(stl_tuple, file):

    v,f = stl_tuple[:2]

    if len(stl_tuple)>2:
        fn = stl_tuple[2]
    else:
        v10 = v[f[:,2],:] - v[f[:,0],:]
        v20 = v[f[:,-1],:] - v[f[:,0],:]
        fn = np.cross(v10, v20)
        fn = fn / np.sum(fn**2, axis=1)[:,None]**.5
                
    data = np.hstack((fn, v[f[:,0]], v[f[:,1]], v[f[:,2]])).tolist() # to write in single precision
    bs = bytearray(80)
    bs += struct.pack('I', len(data))
    bs += b''.join( [struct.pack('f'*len(d), *d) + b'\x00\x00' for d in data] )

    with open(file, 'wb') as f:
        f.write(bs)

    return None


