# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = None
        self.anchor_class_ids = None
        self.anchor_class_names = []
        self.anchor_pos_thresholds = None
        self.anchor_neg_thresholds = None
        self.class_name_to_id = {}
        self._anchor_cache = None

    def _ensure_anchor_spec(self):
        if self.anchor_class_ids is None:
            self._build_anchor_spec()

    def _build_anchor_spec(self):
        anchor_args = self.params['anchor_args']
        if 'generator_configs' in anchor_args:
            class_names = anchor_args.get('class_names', [])
            self.class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
            specs = []
            for cfg in anchor_args['generator_configs']:
                class_name = cfg['class_name']
                if class_name not in self.class_name_to_id:
                    raise ValueError(f'Unknown class_name in generator_configs: {class_name}')
                rotations = cfg['anchor_rotations']
                for rot in rotations:
                    specs.append({
                        'class_name': class_name,
                        'class_id': self.class_name_to_id[class_name],
                        'h': float(cfg['h']),
                        'w': float(cfg['w']),
                        'l': float(cfg['l']),
                        'z': float(cfg['z']),
                        'r': math.radians(float(rot)),
                        'matched_threshold': float(cfg['matched_threshold']),
                        'unmatched_threshold': float(cfg['unmatched_threshold']),
                    })
            self.anchor_class_names = class_names
        else:
            rotations = anchor_args['r']
            specs = []
            for rot in rotations:
                specs.append({
                    'class_name': anchor_args.get('class_name', 'car'),
                    'class_id': 0,
                    'h': float(anchor_args['h']),
                    'w': float(anchor_args['w']),
                    'l': float(anchor_args['l']),
                    'z': float(anchor_args.get('z', -1.0)),
                    'r': math.radians(float(rot)),
                    'matched_threshold': float(self.params['target_args']['pos_threshold']),
                    'unmatched_threshold': float(self.params['target_args']['neg_threshold']),
                })
            self.class_name_to_id = {specs[0]['class_name']: 0}
            self.anchor_class_names = [specs[0]['class_name']]

        self.anchor_num = len(specs)
        self.anchor_class_ids = np.asarray([spec['class_id'] for spec in specs], dtype=np.int64)
        self.anchor_pos_thresholds = np.asarray([spec['matched_threshold'] for spec in specs], dtype=np.float32)
        self.anchor_neg_thresholds = np.asarray([spec['unmatched_threshold'] for spec in specs], dtype=np.float32)
        return specs

    def _get_box_filter_limits(self):
        self._ensure_anchor_spec()
        specs = self._build_anchor_spec()
        max_h = max(spec['h'] for spec in specs)
        max_w = max(spec['w'] for spec in specs)
        max_l = max(spec['l'] for spec in specs)
        max_xy = math.sqrt(max_w ** 2 + max_l ** 2) * 1.5
        max_z = max_h * 1.5
        z_min, z_max = self.params['anchor_args']['cav_lidar_range'][2], self.params['anchor_args']['cav_lidar_range'][5]
        z_margin = max_h
        return max_xy, max_z, z_min - z_margin, z_max + z_margin

    def _filter_pred_boxes(self, pred_box3d_tensor):
        if pred_box3d_tensor.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=pred_box3d_tensor.device)

        max_xy, max_z, z_lower, z_upper = self._get_box_filter_limits()
        x_len = torch.max(pred_box3d_tensor[:, :, 0], dim=1)[0] - torch.min(pred_box3d_tensor[:, :, 0], dim=1)[0]
        y_len = torch.max(pred_box3d_tensor[:, :, 1], dim=1)[0] - torch.min(pred_box3d_tensor[:, :, 1], dim=1)[0]
        z_max = torch.max(pred_box3d_tensor[:, :, 2], dim=1)[0]
        z_min = torch.min(pred_box3d_tensor[:, :, 2], dim=1)[0]
        z_len = z_max - z_min

        keep = torch.logical_and(x_len <= max_xy, y_len <= max_xy)
        keep = torch.logical_and(keep, z_len <= max_z)
        keep = torch.logical_and(keep, z_min >= z_lower)
        keep = torch.logical_and(keep, z_max <= z_upper)
        return keep

    def generate_anchor_box(self):
        self._ensure_anchor_spec()
        if self._anchor_cache is not None:
            return self._anchor_cache.copy()

        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']
        vh = self.params['anchor_args']['vh']
        vw = self.params['anchor_args']['vw']
        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]
        feature_stride = self.params['anchor_args'].get('feature_stride', 2)

        specs = self._build_anchor_spec()
        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)
        cx, cy = np.meshgrid(x, y)

        anchors = np.zeros((cx.shape[0], cx.shape[1], self.anchor_num, 7), dtype=np.float32)
        anchors[..., 0] = np.repeat(cx[..., np.newaxis], self.anchor_num, axis=2)
        anchors[..., 1] = np.repeat(cy[..., np.newaxis], self.anchor_num, axis=2)

        for anchor_idx, spec in enumerate(specs):
            anchors[..., anchor_idx, 2] = spec['z']
            anchors[..., anchor_idx, 3] = spec['h']
            anchors[..., anchor_idx, 4] = spec['w']
            anchors[..., anchor_idx, 5] = spec['l']
            anchors[..., anchor_idx, 6] = spec['r']

        if self.params['order'] != 'hwl':
            sys.exit('Unknown bbx order.')

        self._anchor_cache = anchors
        return anchors.copy()

    def generate_label(self, **kwargs):
        self._ensure_anchor_spec()
        assert self.params['order'] == 'hwl', 'Currently Voxel only support hwl bbx order.'
        gt_box_center = kwargs['gt_box_center']
        anchors = kwargs['anchors']
        masks = kwargs['mask']
        gt_box_types = kwargs.get('gt_box_types', None)

        feature_map_shape = anchors.shape[:2]
        anchors = anchors.reshape(-1, 7)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        gt_box_center_valid = gt_box_center[masks == 1]
        if gt_box_types is None:
            gt_box_types = ['car'] * gt_box_center_valid.shape[0]
        gt_box_types = list(gt_box_types)[:gt_box_center_valid.shape[0]]
        if gt_box_center_valid.shape[0] == 0:
            label_dict = {'pos_equal_one': pos_equal_one,
                          'neg_equal_one': neg_equal_one,
                          'targets': targets}
            return label_dict

        gt_class_ids = np.asarray(
            [self.class_name_to_id.get(name, -1) for name in gt_box_types],
            dtype=np.int64,
        )

        gt_box_corner_valid = box_utils.boxes_to_corners_3d(gt_box_center_valid, self.params['order'])
        anchors_corner = box_utils.boxes_to_corners_3d(anchors, order=self.params['order'])
        anchors_standup_2d = box_utils.corner2d_to_standup_box(anchors_corner)
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        anchor_class_ids = np.tile(self.anchor_class_ids, feature_map_shape[0] * feature_map_shape[1])
        anchor_pos_thresholds = np.tile(self.anchor_pos_thresholds, feature_map_shape[0] * feature_map_shape[1])
        anchor_neg_thresholds = np.tile(self.anchor_neg_thresholds, feature_map_shape[0] * feature_map_shape[1])
        class_match = anchor_class_ids[:, None] == gt_class_ids[None, :]
        iou = np.where(class_match, iou, -1.0)

        id_highest = np.argmax(iou.T, axis=1)
        id_highest_gt = np.arange(iou.T.shape[0])
        mask_highest = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask_highest], id_highest_gt[mask_highest]

        pos_mask = iou > anchor_pos_thresholds[:, None]
        id_pos, id_pos_gt = np.where(pos_mask)
        neg_mask = np.all(iou < anchor_neg_thresholds[:, None], axis=1)
        id_neg = np.where(neg_mask)[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        targets[index_x, index_y, np.array(index_z) * 7] = (gt_box_center_valid[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = (gt_box_center_valid[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = (gt_box_center_valid[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(gt_box_center_valid[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(gt_box_center_valid[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(gt_box_center_valid[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (gt_box_center_valid[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}
        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = torch.from_numpy(np.array(neg_equal_one))
        targets = torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    def _decode_single_agent(self, cav_content, cav_output):
        self._ensure_anchor_spec()
        transformation_matrix = cav_content['transformation_matrix']
        anchor_box = cav_content['anchor_box']
        prob = cav_output['psm']
        prob = F.sigmoid(prob.permute(0, 2, 3, 1)).reshape(1, -1)
        reg = cav_output['rm']
        dir_logits = cav_output.get('dir_cls_preds', None)
        if dir_logits is not None:
            dir_logits = dir_logits.permute(0, 2, 3, 1).contiguous().reshape(1, -1, 2)
        batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
        mask = torch.gt(prob, self.params['target_args']['score_threshold']).view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

        assert batch_box3d.shape[0] == 1
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])
        anchor_ids = torch.arange(prob.shape[1], device=prob.device)[mask[0]]
        dir_labels = None if dir_logits is None else torch.max(dir_logits, dim=-1)[1][mask]

        if len(boxes3d) == 0:
            return None, None, None, None

        if dir_labels is not None:
            top_labels = (boxes3d[..., -1] > 0) ^ (dir_labels.byte() == 1)
            boxes3d[..., -1] += torch.where(top_labels,
                                            torch.tensor(np.pi).type_as(boxes3d),
                                            torch.tensor(0.0).type_as(boxes3d))

        boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])
        projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
        projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
        boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
        anchor_type_ids = torch.remainder(anchor_ids, self.anchor_num)
        class_ids = torch.from_numpy(self.anchor_class_ids).to(anchor_ids.device)[anchor_type_ids]
        return projected_boxes3d, boxes2d_score, scores, class_ids

    def post_process(self, data_dict, output_dict):
        pred_box3d_tensor, scores, _ = self.post_process_with_label(data_dict, output_dict)
        return pred_box3d_tensor, scores

    def post_process_with_label(self, data_dict, output_dict):
        pred_box3d_list = []
        pred_box2d_list = []
        pred_label_list = []

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            projected_boxes3d, boxes2d_score, scores, class_ids = self._decode_single_agent(cav_content, output_dict[cav_id])
            if projected_boxes3d is None:
                continue
            pred_box3d_list.append(projected_boxes3d)
            pred_box2d_list.append(boxes2d_score)
            pred_label_list.append(class_ids)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None, None

        pred_box2d_list = torch.vstack(pred_box2d_list)
        scores = pred_box2d_list[:, -1]
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        pred_labels = torch.cat(pred_label_list, dim=0)

        keep_index = self._filter_pred_boxes(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        pred_labels = pred_labels[keep_index]

        kept_boxes = []
        kept_scores = []
        kept_labels = []
        for class_id in pred_labels.unique(sorted=True):
            class_mask = pred_labels == class_id
            class_boxes = pred_box3d_tensor[class_mask]
            class_scores = scores[class_mask]
            class_keep = box_utils.nms_rotated(class_boxes, class_scores, self.params['nms_thresh'])
            kept_boxes.append(class_boxes[class_keep])
            kept_scores.append(class_scores[class_keep])
            kept_labels.append(pred_labels[class_mask][class_keep])

        pred_box3d_tensor = torch.cat(kept_boxes, dim=0) if kept_boxes else pred_box3d_tensor[:0]
        scores = torch.cat(kept_scores, dim=0) if kept_scores else scores[:0]
        pred_labels = torch.cat(kept_labels, dim=0) if kept_labels else pred_labels[:0]

        mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]
        pred_labels = pred_labels[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0] == pred_labels.shape[0]
        return pred_box3d_tensor, scores, pred_labels

    @staticmethod
    def delta_to_boxes3d(deltas, anchors, channel_swap=True):
        N = deltas.shape[0]
        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        else:
            deltas = deltas.contiguous().view(N, -1, 7)

        boxes3d = torch.zeros_like(deltas)
        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        anchors_reshaped = anchors.view(-1, 7).float()
        anchors_d = torch.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.unsqueeze(0).repeat(N, 1)
        anchors_reshaped = anchors_reshaped.unsqueeze(0).repeat(N, 1, 1)

        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d.unsqueeze(-1)) + anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[..., [3]]) + anchors_reshaped[..., [2]]
        boxes3d[..., [3, 4, 5]] = torch.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]
        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
