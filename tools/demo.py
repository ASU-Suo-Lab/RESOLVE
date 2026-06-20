import argparse
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from visual_utils import image_vis_utils as img
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
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

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/sunlakes_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/suolab/OpenPCDet/data/sunlakes/high_2lidar_cam/lidar_pc_with_ts/rosbag2_2025_12_20-17_55_21/128_128b_label',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    
    # 4 cameras
    parser.add_argument('--axis1_path', type=str, default='/home/suolab/OpenPCDet/data/sunlakes/2026-01-04_18_02_01-export/resource/rosbag2_2025_12_20-17_55_21/axis1_label')
    parser.add_argument('--axis2_path', type=str, default='/home/suolab/OpenPCDet/data/sunlakes/2026-01-04_18_02_01-export/resource/rosbag2_2025_12_20-17_55_21/axis2_label')
    parser.add_argument('--axis3_path', type=str, default='/home/suolab/OpenPCDet/data/sunlakes/2026-01-04_18_02_01-export/resource/rosbag2_2025_12_20-17_55_21/axis3_label')
    parser.add_argument('--axis4_path', type=str, default='/home/suolab/OpenPCDet/data/sunlakes/2026-01-04_18_02_01-export/resource/rosbag2_2025_12_20-17_55_21/axis4_label')
    
    parser.add_argument('--axis1_frame', type=str, default='')
    parser.add_argument('--axis2_frame', type=str, default='')
    parser.add_argument('--axis3_frame', type=str, default='')
    parser.add_argument('--axis4_frame', type=str, default='')

    # calib directory containing axis1_os128b.txt ... axis4_os128b.txt
    parser.add_argument('--calib_dir', type=str, default='/home/suolab/OpenPCDet/data/sunlakes/2026-01-04_18_02_01-export/spatial_results')
    parser.add_argument('--rot_pc_deg', type=float, default=-142.5,
                    help='extra rotation applied to lidar before projection (deg), e.g. -142.5')
    parser.add_argument('--save_dir', type=str, default='./results', help='if set, save projected images')
    parser.add_argument('--score_thr', type=float, default=0.6)
    parser.add_argument('--img_ext', type=str, default='.jpg')
    
    parser.add_argument('--frame_name', type=str, default='1766278468-450370909.bin',
                    help='basename of lidar file (e.g. 000123.bin) to visualize')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def intensity_to_rgb(intensity, cmap_name='jet', vmin=None, vmax=None):
    """
    intensity: [N]
    return: [N, 3], 数值范围 [0,1]
    """
    intensity = np.asarray(intensity).astype(np.float32)

    if vmin is None:
        vmin = np.percentile(intensity, 1)
    if vmax is None:
        vmax = np.percentile(intensity, 99)

    intensity = np.clip(intensity, vmin, vmax)
    intensity_norm = (intensity - vmin) / max(vmax - vmin, 1e-6)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(intensity_norm)[:, :3]   # 只取 RGB，不要 alpha
    return colors
    
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    if args.frame_name:
        names = [Path(x).name for x in demo_dataset.sample_file_list]
        assert args.frame_name in names, f"{args.frame_name} not found in data_path"
        indices = [names.index(args.frame_name)]
    else:
        indices = range(len(demo_dataset))
    
    # load image lists (sorted, index-aligned)
    axis_imgs = {}
    for cam, p in [("axis1", args.axis1_path), ("axis2", args.axis2_path), ("axis3", args.axis3_path), ("axis4", args.axis4_path)]:
        files = sorted(glob.glob(str(Path(p) / f"*{args.img_ext}")))
        axis_imgs[cam] = files

    n = len(demo_dataset)
    assert all(len(axis_imgs[c]) == n for c in axis_imgs), "Image counts must match lidar count!"
    
    def pick_img_path(args, cam):
        cam_dir = Path(getattr(args, f"{cam}_path"))
        cam_frame = getattr(args, f"{cam}_frame", "")
        if cam_frame:
            p = cam_dir / cam_frame
        else:
            return None
        assert p.exists(), f"Missing {cam} image: {p}"
        return str(p)
    
    # load calibs
    calibs = {}
    for cam in ["axis1", "axis2", "axis3", "axis4"]:
        K,_= img.get_cam_kd(str(Path(args.calib_dir) / f"{cam}_os128b.txt"))
        ext = img.get_extrinsic(Path(args.calib_dir) / f"{cam}_os128b.txt")
        Rz4 = np.eye(4, dtype=np.float32)
        Rz4[:3, :3] = img.rotmat_z_deg(args.rot_pc_deg)   # 需要把 rotmat_z_deg 放进 image_vis_utils.py

        lidar2cam = ext @ Rz4.T   # ✅关键：和正确代码一致
        calibs[cam] = {"K": K, "T": lidar2cam}
        
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
    with torch.no_grad():
        for idx in indices:
            data_dict = demo_dataset[idx]
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            pred_boxes  = pred_dicts[0]['pred_boxes']
            pred_scores = pred_dicts[0]['pred_scores']
            pred_labels = pred_dicts[0]['pred_labels']

            mask = pred_scores >= args.score_thr
            pred_boxes  = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            boxes = pred_boxes.detach().cpu().numpy()
            scores = pred_scores.detach().cpu().numpy()
            labels = pred_labels.detach().cpu().numpy()
            
            horizontal_line = V.create_horizontal_line(
                x1=-100.0,
                x2=100.0,
                y=5.0,
                z=-5.0,
                color=(1.0, 0.0, 0.0)   # 红色
            )
            
            points_all = data_dict['points']
            points_xyz = points_all[:, 1:4].detach().cpu().numpy()
            points_intensity = points_all[:, 4].detach().cpu().numpy()

            point_colors = intensity_to_rgb(points_intensity, cmap_name='jet')

            V.interactive_draw_scenes_with_line(
               points=points_xyz,
               ref_boxes=pred_boxes,
               ref_labels=pred_labels,
               point_colors=point_colors,
               line_mode='vertical',   # 或 'vertical'
               x1=0.0,
               x2=0.0,
               y1=-100.0,
               y2=100.0,
               x=5.0,
               y=0.0,
               z=-4.5,
               line_color=(1.0, 0.0, 0.0),
               step=0.2
            )  
            #V.draw_scenes(
            #    points=data_dict['points'][:, 1:],
            #    ref_boxes=pred_boxes,
            #    ref_scores=pred_scores,
            #    ref_labels=pred_labels,
            #    point_colors=point_colors,
            #    pcd_color=(1,0,0),
            #    extra_geometries=[horizontal_line]
            #)

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
                
            # 4-camera projection
            for cam in ["axis1", "axis2", "axis3", "axis4"]:
                img_path = pick_img_path(args, cam)
                if img_path is None:
                    img_path = axis_imgs[cam][idx]
                img_bgr = cv2.imread(img_path)
                assert img_bgr is not None, f"Failed to read image: {img_path}"


                K = calibs[cam]["K"]
                T = calibs[cam]["T"]
                obbs = img.pcdet_boxes_to_o3d_obbs(boxes)
                out_img = img_bgr.copy()
                
                labels_base_is_one = True
                if labels.size > 0:
                    labels_base_is_one = (labels.min() >= 1)
                for obb, lab in zip(obbs, labels):
                    rgb01 = V.label_to_rgb(int(lab), labels_base_is_one, default=(1.0, 1.0, 1.0))
                    color_bgr = img.rgb01_to_bgr255(rgb01) 
                    img.draw_obbs_on_image(out_img, [obb], K, T, color=color_bgr, thickness=3)

               # cv2.imshow(cam, vis)

                if save_dir:
                    out = save_dir / f"{cam}_{idx:06d}.jpg"
                    cv2.imwrite(str(out), out_img)

            #cv2.waitKey(0)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
