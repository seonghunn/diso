import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from . import _C


class DiffMC(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        if dtype == torch.float32:
            mc = _C.CUMCFloat()
        elif dtype == torch.float64:
            mc = _C.CUMCDouble()

        class DMCFunction(Function):
            @staticmethod
            def forward(ctx, grid, deform, isovalue):
                if deform is None:
                    verts, tris = mc.forward(grid, isovalue)
                else:
                    verts, tris = mc.forward(grid, deform, isovalue)
                ctx.isovalue = isovalue
                ctx.save_for_backward(grid, deform)
                return verts, tris

            @staticmethod
            def backward(ctx, adj_verts, adj_faces):
                grid, deform = ctx.saved_tensors
                DMCFunction.forward(ctx, grid, deform, ctx.isovalue)
                adj_grid = torch.zeros_like(grid)
                if deform is None:
                    mc.backward(
                        grid, ctx.isovalue, adj_verts, adj_grid
                    )
                    return adj_grid, None, None, None, None
                else:
                    adj_deform = torch.zeros_like(deform)
                    mc.backward(
                        grid, deform, ctx.isovalue, adj_verts, adj_grid, adj_deform
                    )
                    return adj_grid, adj_deform, None, None, None

        self.func = DMCFunction

    def forward(self, grid, deform=None, isovalue=0.0, normalize=True):
        if grid.min() >= isovalue or grid.max() <= isovalue:
            return torch.zeros((0, 3), dtype=self.dtype, device=grid.device), torch.zeros((0, 3), dtype=torch.int32, device=grid.device)
        dimX, dimY, dimZ = grid.shape
        grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", isovalue+1)
        if deform is not None:
            deform = F.pad(deform, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)
        verts, tris = self.func.apply(grid, deform, isovalue)
        verts = verts - 1
        if normalize:
            verts = verts / (
                torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1
            )
        return verts, tris.long()

class DiffDMC(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        if dtype == torch.float32:
            dmc = _C.CUDMCFloat()
        elif dtype == torch.float64:
            dmc = _C.CUDMCDouble()

        class DDMCFunction(Function):
            @staticmethod
            def forward(ctx, grid, deform, isovalue):
                if deform is None:
                    verts, quads = dmc.forward(grid, isovalue)
                else:
                    verts, quads = dmc.forward(grid, deform, isovalue)
                ctx.isovalue = isovalue
                ctx.save_for_backward(grid, deform)
                return verts, quads

            @staticmethod
            def backward(ctx, adj_verts, adj_faces):
                grid, deform = ctx.saved_tensors
                DDMCFunction.forward(ctx, grid, deform, ctx.isovalue)
                adj_grid = torch.zeros_like(grid)
                if deform is None:
                    dmc.backward(
                        grid, ctx.isovalue, adj_verts, adj_grid
                    )
                    return adj_grid, None, None, None, None
                else:
                    adj_deform = torch.zeros_like(deform)
                    dmc.backward(
                        grid, deform, ctx.isovalue, adj_verts, adj_grid, adj_deform
                    )
                    return adj_grid, adj_deform, None, None, None

        self.func = DDMCFunction


    def is_concave_quad(self, v0, v1, v2, v3):
        # Case 1: divide quad with (v0, v1, v2) and (v0, v2, v3)
        n1_case1 = torch.cross(v1 - v0, v2 - v0, dim=-1)
        n2_case1 = torch.cross(v2 - v0, v3 - v0, dim=-1)

        # Case 2: divide quad with (v0, v1, v3) and (v1, v2, v3)
        n1_case2 = torch.cross(v1 - v0, v3 - v0, dim=-1)
        n2_case2 = torch.cross(v2 - v1, v3 - v1, dim=-1)

        dot_case1 = torch.sum(n1_case1 * n2_case1, dim=-1)
        dot_case2 = torch.sum(n1_case2 * n2_case2, dim=-1)

        is_concave = (dot_case1 < 0) | (dot_case2 < 0)

        return is_concave
    
    def forward(self, grid, deform=None, isovalue=0.0, return_quads=False, normalize=True):
        if grid.min() >= isovalue or grid.max() <= isovalue:
            return torch.zeros((0, 3), dtype=self.dtype, device=grid.device), torch.zeros((0, 4), dtype=torch.int32, device=grid.device)
        dimX, dimY, dimZ = grid.shape
        grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", isovalue+1)
        if deform is not None:
            deform = F.pad(deform, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)
        verts, quads = self.func.apply(grid, deform, isovalue)
        verts = verts - 1
        if normalize:
            verts = verts / (
                torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1
            )
        if return_quads:
            return verts, quads.long()
        else:
            # divide the quad into two triangles maximize the smallest angle within each triangle
            quads = quads.long()
            face_config1 = torch.tensor([[0, 1, 3], [1, 2, 3]])
            face_config2 = torch.tensor([[0, 1, 2], [0, 2, 3]])
            
            
            v0, v1, v2, v3 = torch.unbind(verts[quads], dim=-2)
            is_concave = self.is_concave_quad(v0, v1, v2, v3)
            
            angles1, angles2 = [], []
            for i in range(len(face_config1)):
                v0, v1, v2 = torch.unbind(verts[quads[:, face_config1[i]]], dim=-2)
                cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
                cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
                cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
                angles1.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])
            for i in range(len(face_config2)):
                v0, v1, v2 = torch.unbind(verts[quads[:, face_config2[i]]], dim=-2)
                cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
                cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
                cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
                angles2.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])

            angles1 = torch.stack(angles1, dim=-1)
            angles2 = torch.stack(angles2, dim=-1)

            convex_quads = ~is_concave
            angles1 = torch.max(angles1, dim=1)[0]
            angles2 = torch.max(angles2, dim=1)[0]

            # Concave quads : minimizing min angle
            concave_faces_1 = quads[is_concave][angles1[is_concave] >= angles2[is_concave]]
            concave_faces_2 = quads[is_concave][angles1[is_concave] < angles2[is_concave]]

            faces_concave_1 = concave_faces_1[:, [0, 1, 3, 1, 2, 3]].view(-1, 3)
            faces_concave_2 = concave_faces_2[:, [0, 1, 2, 0, 2, 3]].view(-1, 3)

            # Convex quads : maximizing min angle
            convex_faces_1 = quads[convex_quads][angles1[convex_quads] < angles2[convex_quads]]
            convex_faces_2 = quads[convex_quads][angles1[convex_quads] >= angles2[convex_quads]]

            faces_convex_1 = convex_faces_1[:, [0, 1, 3, 1, 2, 3]].view(-1, 3)
            faces_convex_2 = convex_faces_2[:, [0, 1, 2, 0, 2, 3]].view(-1, 3)

            faces = torch.cat([faces_concave_1, faces_concave_2, faces_convex_1, faces_convex_2], dim=0)

            return verts, faces.long()