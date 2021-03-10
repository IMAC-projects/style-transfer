import pyvista as pv
import numpy as np
from scipy.spatial import KDTree

p = pv.Plotter()
obj1 = pv.read("../output/final_model_cow.obj")
obj2 = pv.read("../assets/meshes/cow.obj")
p.add_mesh(obj1, smooth_shading=True)
p.add_mesh(obj2, smooth_shading=True)
p.show_grid()
p.show()

tree = KDTree(obj2.points)
d, idx = tree.query(obj1.points)
obj1["distances"] = d

print("Distance :")
print(np.mean(d))

p = pv.Plotter()
p.add_mesh(obj1, scalars="distances", smooth_shading=True)
p.add_mesh(obj2, color=True, opacity=0.75, smooth_shading=True)
p.show()