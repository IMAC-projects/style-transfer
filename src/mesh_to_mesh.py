import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

# set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# load the target mesh
trg_obj = os.path.join('meshes/02t.obj')

# we read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)

# verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
# faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# we scale normalize and center the target mesh
# (scale, center) will be used to bring the predicted mesh to its original center and scale
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale

# Load the source mesh.
src_obj = os.path.join('meshes/02ss_02.obj')

# we read the source 3D model using load_obj
Sverts, Sfaces, Saux = load_obj(src_obj)

Sfaces_idx = Sfaces.verts_idx.to(device)
Sverts = Sverts.to(device)

# we're doing the same transformation as above
Scenter = Sverts.mean(0)
Sverts = Sverts - Scenter
Sscale = max(Sverts.abs().max(0)[0])
Sverts = Sverts / Sscale

# We construct a meshes structure for the source and target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
src_mesh = Meshes(verts=[Sverts], faces=[Sfaces_idx])

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

'''
plot_pointcloud(trg_mesh, "Target mesh")
plot_pointcloud(src_mesh, "Source mesh")'''

# we will learn to deform the source mesh by offsetting its vertices
# the shape of the deform parameters is equal to the total number of vertices in src_mesh
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

Niter = 1000
# weight for the chamfer loss
w_chamfer = 1.0 
# weight for mesh edge loss
w_edge = 1.0 
# weight for mesh normal consistency
w_normal = 0.01 
# weight for mesh laplacian smoothing
w_laplacian = 0.1 
# plot period for the losses
plot_period = 250
loop = tqdm(range(Niter))

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

for i in loop:
    # initialize optimizer
    optimizer.zero_grad()
    # deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    # sample 5k points from the surface of each mesh 
    sample_trg = sample_points_from_meshes(trg_mesh, 5000)
    sample_src = sample_points_from_meshes(new_src_mesh, 5000)
    # compare the two sets of pointclouds by computing the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    # and the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    # weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
    # print the losses
    loop.set_description('total_loss = %.6f' % loss)
    # save the losses for plotting
    chamfer_losses.append(loss_chamfer)
    edge_losses.append(loss_edge)
    normal_losses.append(loss_normal)
    laplacian_losses.append(loss_laplacian)
    # plot mesh
    if i % plot_period == 0:
        plot_pointcloud(new_src_mesh, title="iter: %d" % i)
    # optimization step
    loss.backward()
    optimizer.step()

fig = plt.figure(figsize=(13, 5))
ax = fig.gca()
ax.plot(chamfer_losses, label="chamfer loss")
ax.plot(edge_losses, label="edge loss")
ax.plot(normal_losses, label="normal loss")
ax.plot(laplacian_losses, label="laplacian loss")
ax.legend(fontsize="16")
ax.set_xlabel("Iteration", fontsize="16")
ax.set_ylabel("Loss", fontsize="16")
ax.set_title("Loss vs iterations", fontsize="16")

# fetch the verts and faces of the final predicted mesh
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# scale normalize back to the original target size
final_verts = final_verts * scale + center

# store the predicted mesh using save_obj
final_obj = os.path.join('./meshes', 'final_model.obj')
save_obj(final_obj, final_verts, final_faces)
