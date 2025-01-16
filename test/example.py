import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
from diso import DiffMC
from diso import DiffDMC

# Load SDF from file
sdf_file = "./sdf_small.npy"
if not os.path.exists(sdf_file):
    raise FileNotFoundError(f"File {sdf_file} not found!")
sdf_data = np.load(sdf_file)
# Convert SDF to PyTorch tensor
device = "cuda:0"
sdf = torch.tensor(sdf_data, dtype=torch.float32, device=device)
sdf = sdf.requires_grad_(True)  # Enable gradients for backward pass

os.makedirs("out", exist_ok=True)

# Create grid deformation
deform = torch.nn.Parameter(
    torch.rand(
        (sdf.shape[0], sdf.shape[1], sdf.shape[2], 3),
        dtype=torch.float32,
        device=device,
    ),
    requires_grad=True,
)

# Initialize iso-surface extractors
diffmc = DiffMC(dtype=torch.float32)
diffdmc = DiffDMC(dtype=torch.float32)

verts, faces, tris = diffdmc(sdf, None, isovalue=0, normalize = False)
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=tris.cpu().numpy(), process=False)
#mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(), process=False)
mesh.export("out/diso_origin.obj")

# verts, faces = diffmc(sdf, None, isovalue=0)
# mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(), process=False)
# mesh.export("out/diso_mc.obj")


print("forward results saved to out/")
