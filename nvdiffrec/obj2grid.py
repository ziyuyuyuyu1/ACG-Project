import trimesh
import trimesh.voxel
import numpy as np
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kaolin

# Load the OBJ file
mesh = trimesh.load_mesh('/share1/jialuo/12221_Cat_v1_l3.obj')

# Get the bounds of the mesh
min_bound = mesh.bounds[0]
max_bound = mesh.bounds[1]

# Calculate the size of the mesh
size = max_bound - min_bound

# Calculate the required pitch
pitch = max(size) / 127

print(pitch)
# Create a voxel grid
voxels = mesh.voxelized(pitch=pitch)

# Get the voxel data as a 3D numpy array
voxel_data = voxels.matrix

# If the voxel_data is not exactly 128x128x128, you can pad it to the desired size
if voxel_data.shape != (128, 128, 128):
    pad_x = 128 - voxel_data.shape[0]
    pad_y = 128 - voxel_data.shape[1]
    pad_z = 128 - voxel_data.shape[2]
    voxel_data = np.pad(voxel_data, ((0, pad_x), (0, pad_y), (0, pad_z)))

np.save('test.npy', voxel_data)

# Create a 3D figure
fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_axes(Axes3D(fig)) 

# Get the coordinates of the voxels
x, y, z = voxel_data.nonzero()
# print(x)

# The data greater than 0.5 is considered solid
# x, y, z = x[voxel_data[x, y, z] > 0.5], y[voxel_data[x, y, z] > 0.5], z[voxel_data[x, y, z] > 0.5]

# Plot the voxels
ax.scatter(x, y, z, zdir='z', c='black')

ax.view_init(elev=20, azim=30)

ax.set_box_aspect([np.ptp(a) for a in [x,y,z]])

plt.savefig('test.png')

# Convert the coordinates to float
x, y, z = x.astype(float), y.astype(float), z.astype(float)

# Normalize the coordinates to the range -0.5 to 0.5
x = x / 128 - 0.5
y = y / 128 - 0.5
z = z / 128 - 0.5

# Stack the coordinates to get the point cloud
point_cloud = np.stack([x, y, z], axis=-1)

# np.save('test.npy', point_cloud)
print(point_cloud)