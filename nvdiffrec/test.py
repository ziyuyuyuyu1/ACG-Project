import numpy as np
import mcubes

  # Create a data volume (30 x 30 x 30)
u = np.load('test.npy')
  # Extract the 0-isosurface
print(u.shape)
smoothed_sphere = mcubes.smooth(u)
vertices, triangles = mcubes.marching_cubes(u, 0.8)

print(vertices.shape)
print(triangles.shape)
mcubes.export_obj(vertices, triangles, 'sphere.obj')