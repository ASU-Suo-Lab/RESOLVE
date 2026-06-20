import argparse
import glob
from pathlib import Path
import math
import types

# --- headless-safe matplotlib ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except Exception:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch
import torch_scatter
from pcdet.utils.spconv_utils import spconv

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# ---------------- Dataset ----------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                         root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


# ---------------- Config ----------------
def parse_config():
    parser = argparse.ArgumentParser("VoxSeT: visualize ROI scatter colored by latent activation heat (centered)")

    parser.add_argument('--cfg_file', type=str, default='cfgs/sunlakes_models/voxset.yaml')
    parser.add_argument('--data_path', type=str, default='/home/suolab/OpenPCDet/data/sunlakes/low_2lidar_cam/lidar_pc_with_ts/rosbag2_2025_10_14-07_21_21/16_16b_label')
    parser.add_argument('--ckpt', type=str, default='/home/suolab/OpenPCDet/output/sunlakes_models/voxset/v03_low_lidar_b4_e20/ckpt/checkpoint_epoch_20.pth')
    parser.add_argument('--ext', type=str, default='.bin')

    parser.add_argument('--frame_name', type=str, default='1760451628-450043534.bin')
    parser.add_argument('--save_dir', type=str, default='./results_roi_heat')
    parser.add_argument('--score_thr', type=float, default=0.6)

    # visualization controls
    parser.add_argument('--latent_ids', type=str, default='0,1,2,3')
    parser.add_argument('--norm', type=str, default='p99', choices=['minmax', 'p99', 'p995'])
    parser.add_argument('--point_size', type=float, default=1.0)
    parser.add_argument('--xy_range', type=str, default='', help="optional crop AFTER ROI+centering: 'xmin,xmax,ymin,ymax' (meters)")

    # ROI controls
    parser.add_argument('--roi_mode', type=str, default='id', choices=['topscore', 'first', 'id'])
    parser.add_argument('--roi_box_id', type=int, default=1)
    parser.add_argument('--roi_margin', type=float, default=20, help="expand box dx/dy by +margin meters (context)")

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


# ---------------- Monkey patch: save voxel heat on last VSA layer ----------------
def enable_voxset_last_vsa_heat_saving(model, logger):
    """
    Patch VoxSeT vfe.mlp_vsa_layer_3.forward to save:
      - last_vox_heat: [N_vox, K]  (L2 norm of latent features per voxel)
      - last_inverse : [N_pts]     (point->voxel mapping)
    """
    vfe = getattr(model, "vfe", None)
    if vfe is None and hasattr(model, "module"):
        vfe = getattr(model.module, "vfe", None)

    if vfe is None:
        logger.warning("No model.vfe found; skip VoxSeT visualization.")
        return None

    if not hasattr(vfe, "mlp_vsa_layer_3"):
        logger.warning("model.vfe has no mlp_vsa_layer_3; skip VoxSeT visualization.")
        return None

    layer = vfe.mlp_vsa_layer_3

    if hasattr(layer, "_orig_forward"):
        logger.info("VoxSeT last VSA already patched.")
        return layer

    layer._orig_forward = layer.forward

    def forward_with_save(self, inp, inverse, coords, bev_shape):
        x = self.pre_mlp(inp)

        # encoder: point->latent weights (voxel-internal assignment)
        attn = torch_scatter.scatter_softmax(self.score(x), inverse, dim=0)  # [N_pts, K]
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim * self.k)
        x_ = torch_scatter.scatter_sum(dot, inverse, dim=0)  # [N_vox, K*dim]

        # voxel latent activation heat
        x_lat = x_.view(-1, self.k, self.dim)         # [N_vox, K, dim]
        heat_vox = torch.norm(x_lat, dim=-1)          # [N_vox, K]
        self.last_vox_heat = heat_vox.detach()
        self.last_inverse = inverse.detach()

        # conv ffn (same as original)
        batch_size = int(coords[:, 0].max() + 1)
        h = spconv.SparseConvTensor(torch.relu(x_), coords.int(), bev_shape, batch_size).dense().squeeze(-1)
        h = self.conv_ffn(h).permute(0, 2, 3, 1).contiguous().view(-1, self.conv_dim)

        flatten_indices = coords[:, 0] * bev_shape[0] * bev_shape[1] + coords[:, 1] * bev_shape[1] + coords[:, 2]
        h = h[flatten_indices.long(), :]
        h = h[inverse, :]

        # decoder (same)
        hs = self.norm(h.view(-1, self.dim)).view(-1, self.k, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)

        return torch.cat([inp, hs], dim=-1)

    layer.forward = types.MethodType(forward_with_save, layer)
    logger.info("Enabled VoxSeT voxel heat saving (last VSA) by monkey-patching forward.")
    return layer


# ---------------- Helpers ----------------
def _parse_latent_ids(s):
    s = s.strip()
    if not s:
        return [0, 1, 2, 3]
    return [int(x) for x in s.split(',') if x.strip() != ""]


def _parse_xy_range(s):
    s = s.strip()
    if not s:
        return None
    vals = [float(x) for x in s.split(',')]
    assert len(vals) == 4, "xy_range must be 'xmin,xmax,ymin,ymax'"
    return vals


def normalize_weights(w, mode="p99"):
    w = w.astype(np.float32)
    if mode == "minmax":
        mn, mx = float(w.min()), float(w.max())
        return (w - mn) / (mx - mn + 1e-6)
    elif mode == "p99":
        lo = np.percentile(w, 1.0)
        hi = np.percentile(w, 99.0)
        w2 = np.clip(w, lo, hi)
        return (w2 - lo) / (hi - lo + 1e-6)
    else:  # p995
        lo = np.percentile(w, 0.5)
        hi = np.percentile(w, 99.5)
        w2 = np.clip(w, lo, hi)
        return (w2 - lo) / (hi - lo + 1e-6)


def points_in_rotated_box_bev(points_xy, box_any, margin=0.0):
    """
    points_xy: (N,2) torch
    box_any: (...,) torch, at least first 7 dims are [cx,cy,cz,dx,dy,dz,yaw]
    margin: expand dx,dy by +2*margin
    """
    box = box_any.reshape(-1)
    cx, cy, cz, dx, dy, dz, yaw = box[:7]
    dx = dx + 2.0 * margin
    dy = dy + 2.0 * margin

    px = points_xy[:, 0] - cx
    py = points_xy[:, 1] - cy
    c = torch.cos(-yaw)
    s = torch.sin(-yaw)
    rx = c * px - s * py
    ry = s * px + c * py
    return (rx.abs() <= dx / 2) & (ry.abs() <= dy / 2)


def draw_center_box(ax, dx, dy, color='w', linewidth=1.0):
    x = np.array([ dx/2,  dx/2, -dx/2, -dx/2, dx/2], dtype=np.float32)
    y = np.array([ dy/2, -dy/2, -dy/2,  dy/2, dy/2], dtype=np.float32)
    ax.plot(x, y, color=color, linewidth=linewidth)


def plot_roi_scatter(points_xy, weights, out_path, point_size=1.0, norm_mode="p99",
                     title="", fixed_xlim=None, fixed_ylim=None, box_dims=None):
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    w = normalize_weights(weights, norm_mode)

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    sc = ax.scatter(x, y, c=w, s=point_size, cmap="jet")
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.colorbar(sc, fraction=0.046, pad=0.04)

    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim[0], fixed_xlim[1])
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim[0], fixed_ylim[1])

    #if box_dims is not None:
    #    draw_center_box(ax, box_dims[0], box_dims[1], color='w', linewidth=1.0)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_roi_grid(points_xy, heat_NK, latent_ids, out_path,
                      point_size=1.0, norm_mode="p99",
                      suptitle="", fixed_xlim=None, fixed_ylim=None,
                      ncols=4):
    """
    Make a grid of latent heat scatters with a shared colorbar.
    Default layout: 2 rows x 4 cols (for 8 latents). If latent_ids count != 8,
    rows will be auto computed by ceil(Kshow/ncols).
    """
    Kshow = len(latent_ids)
    nrows = int(math.ceil(Kshow / ncols))

    # ---- shared vmin/vmax across selected codes ----
    vals = np.concatenate([heat_NK[:, k].astype(np.float32) for k in latent_ids], axis=0)
    if norm_mode == "minmax":
        vmin, vmax = float(vals.min()), float(vals.max())
    elif norm_mode == "p99":
        vmin, vmax = float(np.percentile(vals, 1.0)), float(np.percentile(vals, 99.0))
    else:  # p995
        vmin, vmax = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("jet")

    # ---- GridSpec: nrows x (ncols + 1 colorbar column) ----
    fig = plt.figure(figsize=(5 * ncols + 0.8, 5 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols + 1,
        width_ratios=[1] * ncols + [0.05],  # last col for colorbar
        wspace=0.05, hspace=0.10
    )

    # ---- plot each latent in (r,c) ----
    for i, k in enumerate(latent_ids):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])

        w = heat_NK[:, k].astype(np.float32)
        w = np.clip(w, vmin, vmax)

        ax.scatter(points_xy[:, 0], points_xy[:, 1],
                   c=w, s=point_size, cmap=cmap, norm=norm)

        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(f"Latent code {k}")

        if fixed_xlim is not None:
            ax.set_xlim(fixed_xlim[0], fixed_xlim[1])
        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim[0], fixed_ylim[1])

    # ---- hide unused axes (if Kshow < nrows*ncols) ----
    for j in range(Kshow, nrows * ncols):
        r = j // ncols
        c = j % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.axis('off')

    # ---- one shared colorbar spanning all rows ----
    cax = fig.add_subplot(gs[:, -1])   # span all rows in last column
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)

    if suptitle:
        fig.suptitle(suptitle)

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
# ---------------- Main ----------------
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("---- VoxSeT ROI latent-heat visualization (centered) ----")

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    names = [Path(x).name for x in demo_dataset.sample_file_list]
    assert args.frame_name in names, f"{args.frame_name} not found in {args.data_path}"
    idx = names.index(args.frame_name)
    logger.info(f"Using frame: {args.frame_name} (index={idx})")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    vox_layer = enable_voxset_last_vsa_heat_saving(model, logger)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dict = demo_dataset[idx]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)

    pred = pred_dicts[0]
    boxes = pred.get('pred_boxes', None)
    scores = pred.get('pred_scores', None)
    labels = pred.get('pred_labels', None)

    if boxes is None or scores is None or boxes.shape[0] == 0:
        raise RuntimeError("No predicted boxes found in pred_dicts[0].")

    keep = scores >= args.score_thr
    boxes_k = boxes[keep]
    scores_k = scores[keep]
    labels_k = labels[keep] if labels is not None else None

    if boxes_k.shape[0] == 0:
        raise RuntimeError("No boxes after applying score_thr.")

    if args.roi_mode == 'topscore':
        sel = int(torch.argmax(scores_k).item())
    elif args.roi_mode == 'first':
        sel = 0
    else:  # id
        sel = int(np.clip(args.roi_box_id, 0, boxes_k.shape[0] - 1))

    box_sel = boxes_k[sel]   # (>=7,)
    logger.info(f"Selected ROI box idx={sel}, score={float(scores_k[sel]):.3f}, label={int(labels_k[sel]) if labels_k is not None else -1}")

    # Optional: show full scene in Open3D/Mayavi
    try:
        V.draw_scenes(
            points=data_dict['points'][:, 1:],
            ref_boxes=boxes_k,
            ref_scores=scores_k,
            ref_labels=labels_k,
            pcd_color=(0, 1, 0)
        )
        if not OPEN3D_FLAG:
            mlab.show(stop=True)
    except Exception as e:
        logger.warning(f"Draw scenes skipped: {e}")

    # points
    pts = data_dict['points']  # (N,C) torch, col0=batch_idx
    points_xyz = pts[:, 1:4]   # (N,3)
    points_xy = points_xyz[:, :2]

    # voxel heat backfilled to points
    if vox_layer is None or not hasattr(vox_layer, "last_vox_heat"):
        raise RuntimeError("VoxSeT voxel heat not found. Patch failed or model is not VoxSeT.")

    heat_vox = vox_layer.last_vox_heat   # (N_vox, K)
    inv = vox_layer.last_inverse         # (N_pts,)
    heat_pt = heat_vox[inv]              # (N_pts, K)

    # ROI mask
    roi_mask = points_in_rotated_box_bev(points_xy, box_sel, margin=args.roi_margin)

    points_xy_np = points_xy[roi_mask].detach().cpu().numpy()       # (N_roi,2) global
    heat_np = heat_pt[roi_mask].detach().cpu().numpy()              # (N_roi,K)

    # center to box center
    cx = float(box_sel.reshape(-1)[0].item())
    cy = float(box_sel.reshape(-1)[1].item())
    points_xy_np = points_xy_np.copy()
    points_xy_np[:, 0] -= cx
    points_xy_np[:, 1] -= cy

    # fixed view range from box dims + margin
    dx = float(box_sel.reshape(-1)[3].item())
    dy = float(box_sel.reshape(-1)[4].item())
    half_x = dx / 2.0 + args.roi_margin
    half_y = dy / 2.0 + args.roi_margin
    fixed_xlim = (-half_x, half_x)
    fixed_ylim = (-half_y, half_y)

    # optional additional crop after centering
    xy_range = _parse_xy_range(args.xy_range)
    if xy_range is not None:
        xmin, xmax, ymin, ymax = xy_range
        x = points_xy_np[:, 0]
        y = points_xy_np[:, 1]
        m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        points_xy_np = points_xy_np[m]
        heat_np = heat_np[m]

    latent_ids = _parse_latent_ids(args.latent_ids)

    # per-code images
    for k in latent_ids:
        out_path = str(save_dir / f"{args.frame_name}_ROI_heat_code{k}.png")
        plot_roi_scatter(
            points_xy=points_xy_np,
            weights=heat_np[:, k],
            out_path=out_path,
            point_size=args.point_size,
            norm_mode=args.norm,
            #title=f"{args.frame_name} ROI heat code{k} (centered, margin={args.roi_margin})",
            fixed_xlim=fixed_xlim,
            fixed_ylim=fixed_ylim,
            box_dims=(dx, dy)
        )
        logger.info(f"Saved: {out_path}")

    # grid
    grid_path = str(save_dir / f"{args.frame_name}_ROI_heat_grid.png")
    plot_roi_grid(
    points_xy=points_xy_np,
    heat_NK=heat_np,
    latent_ids=latent_ids,      # 建议这里传 8 个：0..7
    out_path=grid_path,
    point_size=args.point_size,
    norm_mode=args.norm,
    fixed_xlim=fixed_xlim,
    fixed_ylim=fixed_ylim,
    ncols=4
    )
    logger.info(f"Saved: {grid_path}")

    logger.info("Done.")


if __name__ == '__main__':
    main()
