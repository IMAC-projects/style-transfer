import numpy as np
import trimesh
import random

mesh = trimesh.load('./assets/box.obj')

# if(len(mesh.split()) > 0) :
#     mesh = mesh.split()[0]
    
# mesh = trimesh.creation.icosphere(1)

print("is_watertight:", mesh.is_watertight) # is the current mesh watertight?

print("euler_number:", mesh.euler_number)  # what's the euler number for the mesh?
print("volume / convex_hull volume:", mesh.volume / mesh.convex_hull.volume)

# since the mesh is watertight, it means there is a volumetric center of mass which we can set as the origin for our mesh
mesh.vertices -= mesh.center_mass

print("moment_inertia:", mesh.moment_inertia) # what's the moment of inertia for the mesh?

print("faces/vertices count:", len(mesh.faces), "/", len(mesh.vertices))

print("subdivision")
# v, f = trimesh.remesh.subdivide(vertices=mesh.vertices, faces=mesh.faces)

max_edge = mesh.scale / 10
v, f = trimesh.remesh.subdivide_to_size(vertices=mesh.vertices, faces=mesh.faces, max_edge=max_edge, max_iter=10, return_index=False)

mesh = trimesh.Trimesh(vertices=v, faces=f)
# mesh = mesh.subdivide()
print("faces/vertices count:", len(mesh.faces), "/", len(mesh.vertices))

mesh.export(file_obj="export.stl", file_type='stl')

# jitter
# for v in mesh.vertices:
#     v[0] += random.random()*0.03
#     v[1] += random.random()*0.03
#     v[2] += random.random()*0.03

for i in range(len(mesh.vertices)):
    mesh.visual.vertex_colors[i] = trimesh.visual.random_color()

# mesh.visual.face_colors = [255, 255, 255, 255] # set the mesh face colors to white
# mesh.visual.vertex_colors = [255, 255, 255, 255] # set the mesh face colors to white

scene = trimesh.Scene([mesh]) # create a scene with both the mesh and the outline edges

scene.show() # preview mesh in an opengl window if you installed pyglet with pip