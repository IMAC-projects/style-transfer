import numpy as np
import trimesh
import random

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', help="increase output verbosity", action="store_true")
    parser.add_argument('-i','--input', help='Input file name', required=True)
    parser.add_argument('-r','--ratio', help='Subdivide ration: bigger mean smaller edges', default=10)
    parser.add_argument('-o','--output', help='Output file name')

    args = parser.parse_args()
    V = args.verbosity

    mesh = trimesh.load(args.input)

    if(V):
        print("is_watertight:", mesh.is_watertight) # is the current mesh watertight?
        print("euler_number:", mesh.euler_number)  # what's the euler number for the mesh?
        print("volume / convex_hull volume:", mesh.volume / mesh.convex_hull.volume)

    # since the mesh is watertight, it means there is a volumetric center of mass which we can set as the origin for our mesh
    mesh.vertices -= mesh.center_mass

    if(V):
        print("moment_inertia:", mesh.moment_inertia) # what's the moment of inertia for the mesh?
        print("faces/vertices count:", len(mesh.faces), "/", len(mesh.vertices))

    if(V):
        print(f"subdivision using ration of {args.ration}")

    max_edge = mesh.scale / args.ratio
    v, f = trimesh.remesh.subdivide_to_size(vertices=mesh.vertices, faces=mesh.faces, max_edge=max_edge, max_iter=10, return_index=False)

    mesh = trimesh.Trimesh(vertices=v, faces=f)

    if(V):
        print("faces/vertices count:", len(mesh.faces), "/", len(mesh.vertices))

    if(V):
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

    if(args.output):
        mesh.export(file_obj=args.output, file_type='obj')

