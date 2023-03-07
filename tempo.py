from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d


fig = plt.figure()
ax=fig.gca(projection='3d')

for i in ["x","y","z"]:
    circle = Circle((0, 0), 1)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)


ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)

plt.show()



import numpy as np

N_LIGA = 20
# figure, axis equal off tight, hold on,
so = object()
so.V, so.F, so.FN = stl_read('B_ROI_R.stl')


def stl_read(cls, file):
if file.endswith('.stl'):
    with open(file, 'rb') as f:
        f.seek(80)
        data = f.read()
    nf, data = struct.unpack('I', data[0:4])[0], data[4:]
    data = struct.unpack('f'*(nf*12), b''.join([data[i*50:i*50+48] for i in range(nf)]))
    data = np.asarray(data).reshape(-1,12)
    FN = data[:,0:3]
    V = data[:,3:12].reshape(-1,3)
    F = np.arange(0,len(V)).reshape(-1,3)
    s = object()
    s.V, s.F, s.FN = V, F, FN
    return s


def remove_duplicate(self):
    if not self.points.size or not self.connectivity.size:
        return
    self.points = self.points.round(decimals=6)
    self.points, ind = np.unique(self.points, axis=0, return_inverse=True)
    self.connectivity = np.unique(ind[self.connectivity], axis=0)
    f_unique, ind = np.unique(self.connectivity, return_inverse=True)
    self.connectivity = ind.reshape(self.connectivity.shape)
    self.points = self.points[f_unique,:]



def ordered_edge(F)
    e = np.vstack( (F[:,0:1], F[:,1:2], F[:,(3,1)]) )
    e = np.sort(e, axis=1)
    eo, ind = np.unique(e, axis=0, return_inverse=True)
    # [eo, ~, ind] = unique(e, 'rows')
    eo = eo[np.bincount(ind)==1,:]
    # eo = eo(histcounts(ind, linspace(.5,max(ind)+.5, max(ind)+1))==1,:)
    ordered_edges_out = eo[0,:].tolist()
    eo = eo[1:,:]
    while len(eo):
        id = np.isin(eo, ordered_edges_out[-1:]).any(axis=1)
        if id.sum() != 1: print('wrong')
        ordered_edges_out += eo[id,:].remove(ordered_edges_out[-1])
        eo = np.delete(eo, id, axis=0)

% patch('Vertices',so.V,'Faces',so.F,'FaceColor','c','EdgeCOlor','k')
plot3(so.V(ordered_edges_out,1),so.V(ordered_edges_out,2),so.V(ordered_edges_out,3),'LineWidth',5)


so = object()
si.V, si.F, si.FN = stl_read('chest_wall_right_p5_2mm_cut_big.stl')
[si.V,si.F] = reorder(si.V,si.F)
e = sort([si.F(:,[1,2]) si.F(:,[2,3]) si.F(:,[3,1])],2)
[ei, ~, ind] = unique(e,'rows')
ei = ei(histcounts(ind, linspace(.5,max(ind)+.5, max(ind)+1))==1,:)
ordered_edges_in = ei(1,:)
ei = ei(2:end,:)
while ~isempty(ei)
[~,id] = ismember(ei, ordered_edges_in(end))
[r,c] = ind2sub(size(id),find(id))
if c == 2
ei(r,[1,2]) = ei(r,[2,1])
end
ordered_edges_in = [ordered_edges_in, ei(r,2)]
ei(r,:) = []
end

% [ei1,~,ind] = unique(ei)
% ind = find(histcounts(ind,.5:max(ind)+.5)~=2)
% scatter3(si.V(ei1(ind),1),si.V(ei1(ind),2),si.V(ei1(ind),3),40,'y','filled')

patch('Vertices',si.V,'Faces',si.F,'FaceColor','c','EdgeCOlor','k')
plot3(si.V(ordered_edges_in,1),si.V(ordered_edges_in,2),si.V(ordered_edges_in,3),'LineWidth',5)

vie = si.V(ordered_edges_in,:)
die = [0,0,0 diff(vie,1,1)]
d = sum(die.^2,2).^.5
dc = cumsum(d)
[~, ind] = min(abs(dc - linspace(dc(1),dc(end),N_LIGA+1)))
liga_in = ordered_edges_in(ind(1:end-1))

scatter3(si.V(liga_in,1),si.V(liga_in,2),si.V(liga_in,3),100,'y','filled')




% center_in = si.V(knnsearch(si.V,mean(si.V)),:)
% center_out = mean(so.V(ordered_edges_out,:))
% center = center_in/2 + center_out/2
% scatter3(center(1),center(2),center(3),100,'r','filled')
% coord = pca(si.V)
% quiver3(center(1),center(2),center(3),coord(1,1)*100,coord(2,1)*100,coord(3,1)*100,'r')
% quiver3(center(1),center(2),center(3),coord(1,2)*100,coord(2,2)*100,coord(3,2)*100,'g')
% quiver3(center(1),center(2),center(3),coord(1,3)*100,coord(2,3)*100,coord(3,3)*100,'b')
