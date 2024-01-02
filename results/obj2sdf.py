import os
import sys
import trimesh
import mesh2sdf
import numpy as np
import time

# filename = './1.obj'

# mesh_scale = 0.8
# size = 64
# level = 2 / size

# mesh = trimesh.load(filename, force='mesh')

# # normalize mesh
# vertices = mesh.vertices
# bbmin = vertices.min(0)
# bbmax = vertices.max(0)
# center = (bbmin + bbmax) * 0.5
# scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
# vertices = (vertices - center) * scale

# # fix mesh
# t0 = time.time()
# sdf, mesh = mesh2sdf.compute(
#     vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
# t1 = time.time()

# # output
# # mesh.vertices = mesh.vertices / scale + center
# # mesh.export(filename[:-4] + '.fixed.obj')
# # np.save(filename[:-4] + '.npy', sdf)
# print(sdf.shape)
# print('It takes %.4f seconds to process %s' % (t1-t0, filename))


import numpy as np
import mcubes

# Run the Marching Cubes algorithm to generate the mesh
sdf = np.load('0.npy')
for i in range(sdf.shape[0]):
    vertices, triangles = mcubes.marching_cubes(sdf[i][0], 0)
    print(vertices.shape, triangles.shape)
    # Export the mesh to an OBJ file
    mcubes.export_obj(vertices, triangles, f"{i}_sdf.obj")