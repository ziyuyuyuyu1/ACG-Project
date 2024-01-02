import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mcubes
import torch
import torch.nn as nn
import trimesh

samples = np.load('0.npy')
# change samples into a bool array
# print(samples[0][0])
# samples = samples > 0.5

bs = samples.shape[0]
print(samples.shape)
for i in range(bs):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_axes(Axes3D(fig)) 

    # Get the coordinates of the voxels, the voxels are the positions with value greater than 0.5
    x, y, z = samples[i, 0].nonzero()
    # x, y, z = x[samples[i, 0, x, y, z] > 0.5], y[samples[i, 0, x, y, z] > 0.5], z[samples[i, 0, x, y, z] > 0.5]
    # for d in range(len(x)):
    #     print(f"{i}:  {x[d]} {y[d]} {z[d]}")
    # u = mcubes.smooth(samples[i][0])
    # samples[i][0] = mcubes.smooth(samples[i][0])
    vertices, triangles = mcubes.marching_cubes(samples[i][0], 0.5)
    print(vertices.shape, triangles.shape)
    # mcubes.export_obj(vertices, triangles, f'{i}.obj')
    # Create a trimesh.Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # # Compute the vertex normals
    # mesh.vertex_normals

    # Save the mesh to an OBJ file with normals
    mesh.export(f'{i}_voxel.obj')
    
    # Plot the voxels
    ax.scatter(x, y, z, zdir='z', c='black')

    ax.view_init(elev=20, azim=30)

    ax.set_box_aspect([np.ptp(a) for a in [x,y,z]])
    plt.savefig(f'{i}.png')
    plt.close()
    
# def read_header(fp):
#     """ Read binvox header. Mostly meant for internal use.
#     """
#     line = fp.readline().strip()
#     if not line.startswith(b'#binvox'):
#         raise IOError('Not a binvox file')
#     dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
#     translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
#     scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
#     line = fp.readline()
#     return dims, translate, scale
# with torch.no_grad():
#     with open('/share1/jialuo/car/02958343/bd33b9efe58ebefa99f8616cdb5dd27c/models/model_normalized.solid.binvox', 'rb') as f:
#         dims, translate, scale = read_header(f)
#         raw_data = np.frombuffer(f.read(), dtype=np.uint8)
#         # if just using reshape() on the raw data:
#         # indexing the array as array[i,j,k], the indices map into the
#         # coords as:
#         # i -> x
#         # j -> z
#         # k -> y
#         # if fix_coords is true, then data is rearranged so that
#         # mapping is
#         # i -> x
#         # j -> y
#         # k -> z
#         values, counts = raw_data[::2], raw_data[1::2]
#         data = np.repeat(values, counts).astype(bool)
#         data = data.reshape(dims)
#         data = np.transpose(data, (0, 2, 1))
#         datum = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        
#     avg_pool = nn.AvgPool3d(2)
#     datum = avg_pool(datum)

# if datum.size(-1) < 64: 
#     diff = 64 - datum.size(-1)
#     datum = torch.nn.functional.pad(datum, (0, diff, 0, diff, 0, diff, 0, 0))
    
# datum = datum.squeeze(0).numpy()
# datum = datum > 0.5
# test_data = mcubes.smooth(datum)
# vertices, triangles = mcubes.marching_cubes(test_data, 0)
# test_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
# test_mesh.export(f'test_voxel.obj')
# fig = plt.figure()
# # ax = fig.gca(projection='3d')
# ax = fig.add_axes(Axes3D(fig)) 

# # Get the coordinates of the voxels, the voxels are the positions with value greater than 0.5
# x, y, z = datum.nonzero()
# for d in range(len(x)):
#     print(f"{i}:  {x[d]} {y[d]} {z[d]}")
# vertices, triangles = mcubes.marching_cubes(datum, 0)
# print(vertices.shape, triangles.shape)
# mcubes.export_obj(vertices, triangles, f'test.obj')

# # Plot the voxels
# ax.scatter(x, y, z, zdir='z', c='black')

# ax.view_init(elev=20, azim=30)

# ax.set_box_aspect([np.ptp(a) for a in [x,y,z]])
# plt.savefig(f'test.png')
# plt.close()