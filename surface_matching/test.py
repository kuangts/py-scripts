import numpy as np
import register

f = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
f = np.array([[0,1,2],[0,1,3]])

register.detect_edges(v,f)
register.define_cutoff(v,f)

