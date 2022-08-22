import numpy as np
import open3d as o3d


# # v = pptk.viewer(pts)
vis_cache = np.load('./vis_cache.npy', allow_pickle=True).item()
ID = 19

# lx, ly, lz = [126,117,0]
l = vis_cache['l'][ID]
lx, ly, lz = l
edge000 = np.array([lx, ly, lz])
edge001 = np.array([lx, ly, lz + 1])
edge010 = np.array([lx, ly + 1, lz])
edge011 = np.array([lx, ly + 1, lz + 1])
edge100 = np.array([lx + 1, ly, lz])
edge101 = np.array([lx + 1, ly, lz + 1])
edge110 = np.array([lx + 1, ly + 1, lz])
edge111 = np.array([lx + 1, ly + 1, lz + 1])

points = np.stack([edge000, edge001, edge010, edge011, edge100, edge101, edge110, edge111])
lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]
colors = [[0, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

sdf000 = vis_cache['sdf000'][ID]
sdf001 = vis_cache['sdf001'][ID]
sdf010 = vis_cache['sdf010'][ID]
sdf011 = vis_cache['sdf011'][ID]
sdf100 = vis_cache['sdf100'][ID]
sdf101 = vis_cache['sdf101'][ID]
sdf110 = vis_cache['sdf110'][ID]
sdf111 = vis_cache['sdf111'][ID]
sdfs = [sdf000, sdf001, sdf010, sdf011, sdf100, sdf101, sdf110, sdf111]


sdfs = np.array([-1, 1, 1, 1, 1, 1, 1, 1])
sdfs = sdfs - sdfs.mean()

sdf000, sdf001, sdf010, sdf011, sdf100, sdf101, sdf110, sdf111 = \
    sdfs


SURFACE_RANGE = 1.25
EP = 1e3

ys = np.linspace(ly - SURFACE_RANGE, ly + SURFACE_RANGE, int(EP))
zs = np.linspace(lz - SURFACE_RANGE, lz + SURFACE_RANGE, int(EP))

y_mesh, z_mesh = np.meshgrid(ys,zs)
yz = np.vstack([y_mesh.flatten(), z_mesh.flatten()]).T

pts = np.zeros([int(EP**2), 3])
pts[:,1:] = yz

wys, wzs = pts[:,1]-ly, pts[:,2]-lz
c00 = sdf000 * (1.-wzs) + sdf001 * wzs
c01 = sdf010 * (1.-wzs) + sdf011 * wzs
c10 = sdf100 * (1.-wzs) + sdf101 * wzs
c11 = sdf110 * (1.-wzs) + sdf111 * wzs
c0 = c00 * (1.-wys) + c01 * wys
c1 = c10 * (1.-wys) + c11 * wys
# sdf = c0 * (1.-wx) + c1 * wx = 0

wxs =  c0 / (c0 - c1)

pts[:,0] = wxs.flatten() + lx

def verify(xyz, lx=lx, ly=ly, lz=lz):
    wx, wy, wz = xyz[0] - lx, xyz[1] - ly, xyz[2] - lz
    c00 = sdf000 * (1.-wz) + sdf001 * wz
    c01 = sdf010 * (1.-wz) + sdf011 * wz
    c10 = sdf100 * (1.-wz) + sdf101 * wz
    c11 = sdf110 * (1.-wz) + sdf111 * wz
    c0 = c00 * (1.-wy) + c01 * wy
    c1 = c10 * (1.-wy) + c11 * wy
    sdf = c0 * (1.-wx) + c1 * wx
    return sdf, c0 / (c0 - c1)

# filter out points that are too far away
near_pt_ids = np.arange(pts.shape[0])[np.abs(pts[:, 0] - lx) <= 4*SURFACE_RANGE]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts[near_pt_ids,:])

# Camera line
origin = vis_cache['origins'][ID]
viewdir = vis_cache['dirs'][ID]
t1 = max(vis_cache['t'][ID] - 5, 0)
t2 = vis_cache['t'][ID] + 5

camera_ray = o3d.geometry.LineSet()
camera_ray.points = o3d.utility.Vector3dVector([origin + t1 * viewdir, origin + t2 * viewdir])
camera_ray.lines = o3d.utility.Vector2iVector([[0,1]])
camera_ray.colors = o3d.utility.Vector3dVector([[1,0,0]])

print(f'Camera origin: {origin}')
print(f'Camera view dir: {viewdir}')
print(f'SDFs: {sdfs}')
print(f'Voxel coords: {[lx,ly,lz]}')
print(f'Is surface inside voxel: {((pts >= l).all(axis=-1) & (pts <= l+1).all(axis=-1)).any()}')

geometries = [line_set, pcd, camera_ray]
# geometries = [line_set]
viewer = o3d.visualization.Visualizer()
viewer.create_window()
for geometry in geometries:
    viewer.add_geometry(geometry)
opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()





# o3d.visualization.draw_geometries([pts])