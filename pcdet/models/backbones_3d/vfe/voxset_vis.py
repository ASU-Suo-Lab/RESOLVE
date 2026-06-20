import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import math

from .vfe_template import VFETemplate
from ....utils.spconv_utils import spconv


class MLP(nn.Module):
    """Very simple multi-layer perceptron (FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VoxSeT(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, grid_size, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.num_latents = self.model_cfg.NUM_LATENTS
        self.input_dim = self.model_cfg.INPUT_DIM
        self.output_dim = self.model_cfg.OUTPUT_DIM

        self.input_embed = MLP(num_point_features, 16, self.input_dim, 2)

        self.pe0 = PositionalEncodingFourier(64, self.input_dim)
        self.pe1 = PositionalEncodingFourier(64, self.input_dim * 2)
        self.pe2 = PositionalEncodingFourier(64, self.input_dim * 4)
        self.pe3 = PositionalEncodingFourier(64, self.input_dim * 8)

        self.mlp_vsa_layer_0 = MLP_VSA_Layer(self.input_dim * 1, self.num_latents[0])
        self.mlp_vsa_layer_1 = MLP_VSA_Layer(self.input_dim * 2, self.num_latents[1])
        self.mlp_vsa_layer_2 = MLP_VSA_Layer(self.input_dim * 4, self.num_latents[2])
        self.mlp_vsa_layer_3 = MLP_VSA_Layer(self.input_dim * 8, self.num_latents[3])

        # ---------------------- SAVE FLAGS (你可以改这里) ----------------------
        # 体素级 heat：x_ 聚合到 voxel 后 (N_vox,K,dim) -> norm -> (N_vox,K)
        self.mlp_vsa_layer_3.save_heat = True

        # 点级 heat：每个点一个值 (N_pts,K)
        self.mlp_vsa_layer_3.save_point = True
        # 可选：'attn'|'energy'|'attn_energy'|'logits_relu'|'logits_sigmoid'|'logits_softmax'
        self.mlp_vsa_layer_3.point_mode = "attn_energy"
        # softmax 温度（仅 logits_softmax 用）
        self.mlp_vsa_layer_3.softmax_temp = 1.0
        # ---------------------------------------------------------------------

        self.post_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 16, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01)
        )

        # buffers
        self.register_buffer('point_cloud_range', torch.FloatTensor(point_cloud_range).view(1, -1))
        self.register_buffer('voxel_size', torch.FloatTensor(voxel_size).view(1, -1))
        self.grid_size = grid_size.tolist()

        a, b, c = voxel_size
        self.register_buffer('voxel_size_02x', torch.FloatTensor([a * 2, b * 2, c]).view(1, -1))
        self.register_buffer('voxel_size_04x', torch.FloatTensor([a * 4, b * 4, c]).view(1, -1))
        self.register_buffer('voxel_size_08x', torch.FloatTensor([a * 8, b * 8, c]).view(1, -1))

        a, b, c = grid_size
        self.grid_size_02x = [a // 2, b // 2, c]
        self.grid_size_04x = [a // 4, b // 4, c]
        self.grid_size_08x = [a // 8, b // 8, c]

    def get_output_feature_dim(self):
        return self.output_dim

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (N, C), col0=batch_idx
        points_offsets = points[:, 1:4] - self.point_cloud_range[:, :3]

        min_tensor = points.new_tensor([0, 0, 0])

        # 01x
        coords01x = points[:, :4].clone()
        coords01x[:, 1:4] = points_offsets // self.voxel_size
        coords01x[:, 1:4] = torch.clamp(coords01x[:, 1:4], min_tensor, points.new_tensor(self.grid_size) - 1)
        pe_raw = (points_offsets - coords01x[:, 1:4] * self.voxel_size) / self.voxel_size
        coords01x, inverse01x = torch.unique(coords01x, return_inverse=True, dim=0)

        # 02x
        coords02x = points[:, :4].clone()
        coords02x[:, 1:4] = points_offsets // self.voxel_size_02x
        coords02x[:, 1:4] = torch.clamp(coords02x[:, 1:4], min_tensor, points.new_tensor(self.grid_size_02x) - 1)
        coords02x, inverse02x = torch.unique(coords02x, return_inverse=True, dim=0)

        # 04x
        coords04x = points[:, :4].clone()
        coords04x[:, 1:4] = points_offsets // self.voxel_size_04x
        coords04x[:, 1:4] = torch.clamp(coords04x[:, 1:4], min_tensor, points.new_tensor(self.grid_size_04x) - 1)
        coords04x, inverse04x = torch.unique(coords04x, return_inverse=True, dim=0)

        # 08x
        coords08x = points[:, :4].clone()
        coords08x[:, 1:4] = points_offsets // self.voxel_size_08x
        coords08x[:, 1:4] = torch.clamp(coords08x[:, 1:4], min_tensor, points.new_tensor(self.grid_size_08x) - 1)
        coords08x, inverse08x = torch.unique(coords08x, return_inverse=True, dim=0)

        # embed
        src = self.input_embed(points[:, 1:])  # (N, input_dim)

        # VSA stages
        src = src + self.pe0(pe_raw)
        src = self.mlp_vsa_layer_0(src, inverse01x, coords01x, self.grid_size)

        src = src + self.pe1(pe_raw)
        src = self.mlp_vsa_layer_1(src, inverse02x, coords02x, self.grid_size_02x)

        src = src + self.pe2(pe_raw)
        src = self.mlp_vsa_layer_2(src, inverse04x, coords04x, self.grid_size_04x)

        src = src + self.pe3(pe_raw)
        src = self.mlp_vsa_layer_3(src, inverse08x, coords08x, self.grid_size_08x)

        # ---- EXPORT for visualization ----
        if getattr(self.mlp_vsa_layer_3, "last_point_heat", None) is not None:
            batch_dict["vsa_last_point_heat"] = self.mlp_vsa_layer_3.last_point_heat  # (N_pts,K)

        if getattr(self.mlp_vsa_layer_3, "last_vox_heat", None) is not None:
            batch_dict["vsa_last_vox_heat"] = self.mlp_vsa_layer_3.last_vox_heat  # (N_vox,K)
        if getattr(self.mlp_vsa_layer_3, "last_inverse", None) is not None:
            batch_dict["vsa_last_inverse08x"] = self.mlp_vsa_layer_3.last_inverse  # (N_pts,)
        # ----------------------------------

        src = self.post_mlp(src)

        batch_dict['point_features'] = F.relu(src)
        batch_dict['point_coords'] = points[:, :4]

        batch_dict['pillar_features'] = F.relu(torch_scatter.scatter_max(src, inverse01x, dim=0)[0])
        batch_dict['voxel_coords'] = coords01x[:, [0, 3, 2, 1]]

        return batch_dict


class MLP_VSA_Layer(nn.Module):
    def __init__(self, dim, n_latents=8):
        super().__init__()
        self.dim = dim
        self.k = n_latents

        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
        )

        self.score = nn.Linear(dim, n_latents)

        # ---- SAVE FLAGS / CACHE ----
        self.save_heat = False
        self.last_vox_heat = None
        self.last_inverse = None

        self.save_point = False
        self.point_mode = "attn_energy"
        self.softmax_temp = 1.0
        self.last_point_heat = None
        # ---------------------------

        conv_dim = dim * self.k
        self.conv_dim = conv_dim

        self.conv_ffn = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 1, 1, bias=False),
        )

        self.norm = nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, inp, inverse, coords, bev_shape):
        x = self.pre_mlp(inp)        # [N_pts, dim]
        logits = self.score(x)       # [N_pts, K]
        attn = torch_scatter.scatter_softmax(logits, inverse, dim=0)  # [N_pts, K]

        # ---- POINT-LEVEL heat ----
        if self.save_point:
            energy = torch.norm(x, dim=-1, keepdim=True)  # [N_pts,1]
            mode = self.point_mode
            T = float(self.softmax_temp)

            if mode == "attn":
                point_heat = attn
            elif mode == "energy":
                point_heat = energy.repeat(1, self.k)
            elif mode == "attn_energy":
                point_heat = attn * energy
            elif mode == "logits_relu":
                point_heat = torch.relu(logits)
            elif mode == "logits_sigmoid":
                point_heat = torch.sigmoid(logits)
            else:  # logits_softmax
                point_heat = torch.softmax(logits / max(T, 1e-6), dim=1)

            self.last_point_heat = point_heat.detach()
        # --------------------------

        # aggregate to voxel representation
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim * self.k)
        x_ = torch_scatter.scatter_sum(dot, inverse, dim=0)  # [N_vox, K*dim]

        # ---- VOXEL-LEVEL heat ----
        if self.save_heat:
            x_lat = x_.view(-1, self.k, self.dim)   # [N_vox, K, dim]
            heat_vox = torch.norm(x_lat, dim=-1)    # [N_vox, K]
            self.last_vox_heat = heat_vox.detach()
            self.last_inverse = inverse.detach()
        # --------------------------

        # conv ffn
        batch_size = int(coords[:, 0].max() + 1)
        h = spconv.SparseConvTensor(F.relu(x_), coords.int(), bev_shape, batch_size).dense().squeeze(-1)
        h = self.conv_ffn(h).permute(0, 2, 3, 1).contiguous().view(-1, self.conv_dim)

        flatten_indices = coords[:, 0] * bev_shape[0] * bev_shape[1] + coords[:, 1] * bev_shape[1] + coords[:, 2]
        h = h[flatten_indices.long(), :]
        h = h[inverse, :]

        # decoder
        hs = self.norm(h.view(-1, self.dim)).view(-1, self.k, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0].view(-1, self.dim)

        return torch.cat([inp, hs], dim=-1)


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len

        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)
        pos = self.token_projection(pos)
        return pos
