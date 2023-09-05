import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
import shapely
from shapely import Point, LineString, intersection, Polygon, MultiPolygon

small = [
    [-1,-1],
    [-1, 1],
    [ 1, 1],
    [ 1,-1],
]
big = [[a[0]*2,a[1]*2] for a in small]
bigger = [[a[0]*3,a[1]*3] for a in small]

small1 = [[a[0],a[1]+10] for a in small]
big1 = [[a[0]*2,a[1]*2+10] for a in small]
bigger1 = [[a[0]*3,a[1]*3+10] for a in small]

vtx = np.array(small + big + bigger + small1 + big1 + bigger1)
edges = np.arange(24).reshape(6,4).tolist()
# edges = []
# edges.append(small)
# edges.append(big)
# edges.append(bigger)
# edges.append(small1)
# edges.append(big1)
# edges.append(bigger1)

edge_relation = np.empty((len(edges), len(edges)))
edge_relation[...] = np.nan

for i in range(len(edges)):
    for j in range(len(edges)):
        if i==j or not np.isnan(edge_relation[i,j]):
            continue
        print(i,j)
        if Polygon(shell=vtx[edges[i]]).contains(Polygon(shell=vtx[edges[j]])):
            edge_relation[i,j] = 1
            edge_relation[j,i] = -1
            edge_relation[edge_relation[i]==-1,j] = 1
            edge_relation[j,edge_relation[i]==-1] = -1

        elif Polygon(shell=vtx[edges[j]]).contains(Polygon(shell=vtx[edges[i]])):
            edge_relation[i,j] = -1
            edge_relation[j,i] = 1
            edge_relation[edge_relation[i]==1,j] = -1
            edge_relation[j,edge_relation[i]==1] = 1

        elif Polygon(shell=vtx[edges[i]]).disjoint(Polygon(shell=vtx[edges[j]])):
            edge_relation[i,j] = 0
            edge_relation[j,i] = 0
            edge_relation[edge_relation[i]==1,j] = 0
            edge_relation[j,edge_relation[i]==1] = 0

print(edge_relation)

levels = np.sum(edge_relation==-1,axis=1)
start = np.nonzero(levels==0)[0]
for i in start:
    level = 1
    isout = level%2==1
    group = np.nonzero(edge_relation[:,i]==-1)[0]
    while np.sum(levels[group] == level):
        g = group[levels[group] == level]
        


vtx = small + big
seg = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
]
hole = [0,0]
point_outside = [-3,-3]

line = LineString([point_outside, [3,3]])
# intersections = line.intersection(MultiPolygon([Polygon(small+small[0:1]),Polygon(big+big[0:1])]))
print(Polygon(shell=big).contains(Polygon(shell=small)))
isin = Polygon(shell=big+big[0:1], holes=[small+small[0:1]]).contains(Point([1.5,1.5]))


def contains(edge, point):
    pass







A = {'vertices': vtx, 'segments':seg, 'holes':[hole]}
B = tr.triangulate(A, 'qpa0.05')
tr.compare(plt, A, B)
plt.show()