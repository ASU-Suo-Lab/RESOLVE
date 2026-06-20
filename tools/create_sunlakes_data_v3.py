import pickle
import argparse
from sunlakes_dataset_utils import *
from nuscenes.utils.geometry_utils import view_points

CAM_TYPES = {
    "axis1_label": "CAM_FRONT",
    "axis3_label": "CAM_FRONT_LEFT",
    "axis4_label": "CAM_FRONT_RIGHT",
    "axis2_label": "CAM_BACK",
}

def create_groundtruth_database(data_root,with_cam_gt,vis_check):
    data_root = Path(data_root) 

    database_save_path = data_root / 'sunlakes_gt_database'
    db_info_save_path = data_root / 'sunlakes_dbinfos_train.pkl'
    database_save_path.mkdir(parents=True, exist_ok=True)

    if with_cam_gt:
         img_database_save_path = data_root / 'sunlakes_img_gt_database'
         img_database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    data_path = data_root / 'sunlakes_infos_train.pkl'
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    for idx in tqdm(range(len(data)), desc="Creating point cloud GT"):
        sample_idx = idx
        info = data[idx]
        lidar_path = info['lidar_path']
        lidar_points = read_points(lidar_path, dim=5)
        gt_boxes = info['gt_boxes']
        gt_names = info['gt_names']

        gt_boxes_copy = gt_boxes.copy()
        gt_boxes_copy[:, 2] -= gt_boxes_copy[:, 5] / 2
        indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
            points_in_bboxes_v2(
                points=lidar_points,
                dimensions=gt_boxes[:, 3:6].astype(np.float32),
                location=gt_boxes_copy[:, 0:3].astype(np.float32),
                rotation_y=gt_boxes[:, 6].astype(np.float32),
                name=gt_names
            )
        if with_cam_gt:
            if gt_boxes.shape[0] == 0:
                continue
            # ====== 断言：检查 2D/3D 对齐、cam_index 合法 ======
            assert 'gt_boxes_2d' in info and 'empty_mask' in info, \
                "with_cam_gt=True 但 info 缺少 gt_boxes_2d 或 empty_mask"

            gt_boxes_2d_full = info['gt_boxes_2d']
            empty_mask = info['empty_mask']

            assert gt_boxes_2d_full.shape[0] == empty_mask.shape[0], \
                f"gt_boxes_2d_full({gt_boxes_2d_full.shape[0]}) != empty_mask({empty_mask.shape[0]})"

            assert int(empty_mask.sum()) == gt_boxes.shape[0], \
                f"empty_mask.sum({int(empty_mask.sum())}) != gt_boxes({gt_boxes.shape[0]})"

            # 过滤后 2D 框（与 gt_boxes 一一对应）
            gt_boxes_2d_kept = gt_boxes_2d_full[empty_mask]
            assert gt_boxes_2d_kept.shape[0] == gt_boxes.shape[0], \
                f"gt_boxes_2d_kept({gt_boxes_2d_kept.shape[0]}) != gt_boxes({gt_boxes.shape[0]})"

            # cam_index 合法（允许 -1 dummy）
            cam_keys = info.get("cam_keys", list(info["cams"].keys()))
            cam_idx = gt_boxes_2d_kept[:, 4].astype(np.int32)
            assert (cam_idx < len(cam_keys)).all(), \
                f"cam_index 超范围: max={cam_idx.max()}, cams={len(cam_keys)}"
            assert (cam_idx >= -1).all(), \
                f"cam_index 出现 < -1: min={cam_idx.min()}"
            # ====== 按 cam_keys 顺序读图，保持与 cam_index 一致 ======
            images = []
            for ck in cam_keys:
                imgp = info["cams"][ck]["image_paths"]
                img = cv2.imread(str(imgp))
                images.append(img)

            object_img_patches = crop_img_list(images, gt_boxes_2d_kept)

        for i in range(gt_boxes.shape[0]):
            filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
            abs_filepath = os.path.join(database_save_path, filename)
            rel_filepath = os.path.join('sunlakes_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = lidar_points[indices[:, i]]
            gt_points[:, :3] -= gt_boxes[i, :3]
            
            if vis_check:
            	vis_object_points_and_box3d_on_image(
            		info=info,
            		obj_i=i,
            		gt_points_local=gt_points,     # 注意：这里是“已减中心”的局部点
            		cam_key=None,                  # 用 2D GT 的 cam_index 自动选相机
            		max_points=5000
        	)
        	
            with open(abs_filepath, 'wb') as f:
                gt_points.tofile(f)
            if with_cam_gt:
                img_filename = '%s_%s_%d.png' % (sample_idx, gt_names[i], i)
                img_filepath = img_database_save_path / img_filename
                cv2.imwrite(str(img_filepath),object_img_patches[i])

            db_info = {
                'name': gt_names[i],
                'path': rel_filepath,
                'image_idx': sample_idx,
                'gt_idx': i,
                'box3d_lidar': gt_boxes[i],
                'num_points_in_gt': gt_points.shape[0]
            }
            if with_cam_gt:
                img_db_path = str(img_filepath.relative_to(data_root))  # gt_database/xxxxx.png                        
                db_info.update({'box2d_camera':gt_boxes_2d_kept[i],'img_path':img_db_path,'img_shape':object_img_patches[i].shape})

            if gt_names[i] in all_db_infos:
                all_db_infos[gt_names[i]].append(db_info)
            else:
                all_db_infos[gt_names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def create_dataset_from_json(data_root, data_prefix, rot_pc, sensor_type,calib_dir,
                                         use_lidar, use_cam, vis_check):
    data_root = Path(data_root)
    json_path = data_root / data_prefix / 'HSQJN_3.json'
    pc_resource_root = data_root / 'lidar_pc_with_ts'

    with open(json_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    pc_dataset = {}
    # >>> 新增：保存上一帧该目标的 (xy, ts)
    prev_state = {}  # key: (batch_id, original_3d_id) -> (np.array([x, y]), ts)
    # 记录每个 scene 的“第一帧frameIndex”（最小）及其信息
    scene_first = {}  # scene_id -> {'idx': int, 'pc': str}
    # 第一帧回填需要的信息：第一帧的 pc_key 和 obj_id->kept_idx
    first_frame_pending = {}  # scene_id -> {'pc': str, 'id2idx': dict}

    for idx, frame in tqdm(enumerate(frames), total=len(frames), desc=f"Parsing {data_prefix} all frames"):
        if use_lidar:
            pc_rel_path = '/'.join(frame['info'].split('/')[-3:])
            pc_rel_path = adjust_label_segment(pc_rel_path, sensor_type)
            pc_id = Path(pc_rel_path).stem
            lidar_path = pc_resource_root / pc_rel_path
            if not os.path.exists(lidar_path):
                print(f"Warning: {lidar_path} 不存在，已跳过")
                continue
            lidar_points = read_points(str(lidar_path), dim=5)

            box3d_labels = [label for label in frame['labels'] if label['drawType'] == 'box3d']
            if not box3d_labels:
                continue
            # 原始 3D id 列表（与 box3d_labels 顺序一致）
            box3d_ids = np.array([int(l.get('id', -1)) for l in box3d_labels], dtype=int)
            locs = np.array([l['points'][:3] for l in box3d_labels])
            dims = np.array([l['points'][6:] for l in box3d_labels])
            rots = np.array([l['points'][5] for l in box3d_labels])
            names = np.array([l['label'] for l in box3d_labels], dtype=object)
            names[names == 'van'] = 'barrier'
            names[names == 'construction vehicle'] = 'construction_vehicle'
            names[names == 'golf cart'] = 'traffic_cone'
            theta = np.deg2rad(rot_pc)
            R_z = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            locs = (locs @ R_z.T)
            rots = rots + theta
            bottom_locs = locs.copy();
            bottom_locs[:, 2] -= dims[:, 2] / 2
            num_lidar_pts = get_points_num_in_bbox(lidar_points, dims, bottom_locs, rots, names)
            mask = num_lidar_pts > 0
            if not np.any(mask): continue

            # >>> 新增：计算速度（在 mask 之前按原顺序计算）
            ts = pc_id_to_ts(pc_id)
            vxvy_full = np.zeros((len(box3d_labels), 2), dtype=np.float32)
            batch_id = frame['batchId']  # 你的 scene_id
            for j, oid in enumerate(box3d_ids):
                # 当前中心（旋转后坐标系），只取 xy
                xy = locs[j, :2].astype(np.float32)
                key = (batch_id, int(oid))

                if key in prev_state:
                    prev_xy, prev_ts = prev_state[key]
                    dt = float(ts - prev_ts)
                    if dt > 1e-6 and dt < 2.0:  # 避免除零/跨段过大（可按采样率调整阈值）
                        vx = (xy[0] - prev_xy[0]) / dt
                        vy = (xy[1] - prev_xy[1]) / dt
                        vxvy_full[j, 0] = vx
                        vxvy_full[j, 1] = vy
                    else:
                        # dt异常时设 0；你也可以选择沿用上一次值或 None
                        vxvy_full[j] = 0.0
                # 更新上一状态
                prev_state[key] = (xy, ts)

            gt_boxes_full = np.concatenate([locs, dims, rots[:, None], vxvy_full], axis=1)
            gt_boxes = gt_boxes_full[mask]  # (K,9)
            velocitys = gt_boxes[:, 7:9].astype(np.float32)

            sec, nano = map(int, pc_id.split('-'))
            ts = sec + nano / 1e9

            # >>> 做完mask后，构建 “原3D id → 保留后索引”的映射
            kept_ids = box3d_ids[mask]
            id_to_kept_idx = {int(oid): int(i) for i, oid in enumerate(kept_ids)}
            batch_id = frame['batchId']
            tracking_ids = [batch_id + '_' + str(boxs_id) for boxs_id in kept_ids]

            pc_dataset[pc_rel_path] = {
                'lidar_path': lidar_path,
                'token': pc_id,
                'scene_id': batch_id,
                'tracking_id': tracking_ids,
                'timestamp': ts,
                'gt_boxes': gt_boxes,
                'gt_boxes_velocity': velocitys,
                'gt_names': names[mask],
                'num_lidar_pts': num_lidar_pts[mask],
            }

            pc_dataset[pc_rel_path]['_locs_full_for_cam'] = locs.astype(np.float32)
            pc_dataset[pc_rel_path]['_dims_full_for_cam'] = dims.astype(np.float32)
            pc_dataset[pc_rel_path]['_rots_full_for_cam'] = rots.astype(np.float32)
            pc_dataset[pc_rel_path]['_empty_mask_full_for_cam'] = mask.copy()

            # ==== 下面是基于 frameIndex 的“第一帧复制第二帧速度”逻辑 ====
            fidx = frame.get('frameIndex', None)

            # 1) 发现更早的第一帧，就记录下来并准备回填
            if fidx is not None:
                # 若该 scene 没记录，或遇到更小的 frameIndex，则更新“第一帧”
                if (batch_id not in scene_first) or (fidx < scene_first[batch_id]['idx']):
                    scene_first[batch_id] = {'idx': fidx, 'pc': pc_rel_path}
                    first_frame_pending[batch_id] = {'pc': pc_rel_path, 'id2idx': dict(id_to_kept_idx)}

            # 2) 当遇到“第二帧”（frameIndex == 第一帧 + 1）时，回填第二帧速度到第一帧
            if (fidx is not None) and (batch_id in scene_first):
                first_idx = scene_first[batch_id]['idx']
                if fidx == first_idx + 1 and batch_id in first_frame_pending:
                    prev_pc = first_frame_pending[batch_id]['pc']
                    prev_map = first_frame_pending[batch_id]['id2idx']  # obj_id -> kept_idx (在第一帧)
                    # 遍历当前帧（第二帧）中所有保留目标，将其速度复制到第一帧对应目标
                    for oid, cur_idx in id_to_kept_idx.items():
                        if oid in prev_map:
                            first_idx_in_frame = prev_map[oid]
                            v2 = pc_dataset[pc_rel_path]['gt_boxes_velocity'][cur_idx].astype(np.float32)
                            # 回填到第一帧的独立速度字段
                            pc_dataset[prev_pc]['gt_boxes_velocity'][first_idx_in_frame] = v2
                            # 同步回填到第一帧的 gt_boxes 第 7~8 列
                            pc_dataset[prev_pc]['gt_boxes'][first_idx_in_frame, 7:9] = v2
                    # 仅复制“第二帧”的速度，完成后移除 pending
                    del first_frame_pending[batch_id]


        if use_cam:
            calib_dir = Path(calib_dir) if calib_dir else (data_root / data_prefix/ "spatial_results")
            calib_map = {
                "axis1_label": calib_dir / "axis1_os128b.txt",
                "axis2_label": calib_dir / "axis2_os128b.txt",
                "axis3_label": calib_dir / "axis3_os128b.txt",
                "axis4_label": calib_dir / "axis4_os128b.txt",
            }
            pc_rel_path = '/'.join(frame['info'].split('/')[-3:])
            pc_rel_path = adjust_label_segment(pc_rel_path, sensor_type)
            # 这帧如果没有 lidar entry（上面 continue 掉了），就跳过相机
            if pc_rel_path not in pc_dataset:
                continue

            pc_dataset[pc_rel_path].setdefault("cams", {})
            for img_info in frame.get("imgInfo", []):
                img_rel_path = "/".join(img_info.split("/")[-3:])
                img_name = img_rel_path.split("/")[-2]
                img_path = data_root / data_prefix / "resource" / img_rel_path

                C = calib_map.get(img_name)
                if C is None or not C.exists():
                    # 标定缺失时跳过该相机
                    continue

                cam_key = CAM_TYPES[img_name]
                K, _ = get_cam_kd(C)
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = K

                ext = get_extrinsic(C).astype(np.float32)  # 原始 lidar2cam（4x4）
                Rz4 = rotmat_z(rot_pc, homogeneous=True)  # 旋转（4x4，绕 Lidar 原点）
                lidar2camera = ext @ Rz4.T

                lidar2image = camera_intrinsics @ lidar2camera

                lidar2cam_rot = lidar2camera[:3, :3]
                lidar2cam_trans = lidar2camera[:3, 3:4]
                camera2lidar = np.eye(4)
                camera2lidar[:3, :3] = lidar2cam_rot.T
                camera2lidar[:3, 3:4] = -1 * np.matmul(
                    lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))

                pc_dataset[pc_rel_path]["cams"][cam_key] = {
                    "image_paths": str(img_path),
                    "lidar2camera": lidar2camera,
                    "lidar2image": lidar2image,
                    "camera_intrinsics": camera_intrinsics,
                    "camera2lidar": camera2lidar
                }

                if vis_check:
                    check_lidar_to_img(pc_dataset,pc_rel_path, cam_key)
                    check_img_to_lidar(pc_dataset,pc_rel_path, cam_key)

            # 2) with_cam: 生成 2D GT（需要 full box 信息）
            if not all(k in pc_dataset[pc_rel_path] for k in ['_locs_full_for_cam', '_dims_full_for_cam', '_rots_full_for_cam', '_empty_mask_full_for_cam']):
                continue
            locs_full_for_cam = pc_dataset[pc_rel_path]['_locs_full_for_cam']
            dims_full_for_cam = pc_dataset[pc_rel_path]['_dims_full_for_cam']
            rots_full_for_cam = pc_dataset[pc_rel_path]['_rots_full_for_cam']
            mask_full_for_cam = pc_dataset[pc_rel_path]['_empty_mask_full_for_cam']

            pc_dataset[pc_rel_path]['empty_mask'] = mask_full_for_cam
            gt_boxes_full_before_mask = np.concatenate([locs_full_for_cam, dims_full_for_cam, rots_full_for_cam[:, None]], axis=1).astype(np.float32)  # (N,7)
            corners_lidar = corners_for_projection_from_gtboxes(gt_boxes_full_before_mask[:, :7], yaw_flip=False)

            # 固定相机顺序，保证 cam_index 稳定
            preferred = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"]
            cams_dict = pc_dataset[pc_rel_path]["cams"]
            cam_keys = [k for k in preferred if k in cams_dict]
            pc_dataset[pc_rel_path]["cam_keys"] = cam_keys

            cam_imsize = {}
            for ck in cam_keys:
                imgp = cams_dict[ck]["image_paths"]
                img = cv2.imread(str(imgp))
                if img is None:
                    cam_imsize[ck] = (1600, 900)
                else:
                    h, w = img.shape[:2]
                    cam_imsize[ck] = (w, h)

            gt_boxes_2d = []
            all_2d_boxes = []

            for bi in range(corners_lidar.shape[0]):
                has_proj = False
                for cam_index, cam_key in enumerate(cam_keys):
                    cam = cams_dict[cam_key]
                    K3 = cam["camera_intrinsics"][:3, :3].astype(np.float32)
                    lidar2cam = cam["lidar2camera"].astype(np.float32)

                    pts = corners_lidar[bi]  # (3,8)
                    pts_h = np.vstack([pts, np.ones((1, pts.shape[1]), dtype=np.float32)])  # (4,8)
                    pts_cam = (lidar2cam @ pts_h)[:3, :]  # (3,8)

                    in_front = np.argwhere(pts_cam[2, :] > 0).flatten()
                    if in_front.size == 0:
                        continue
                    pts_cam_front = pts_cam[:, in_front]

                    corner_coords = view_points(pts_cam_front, K3, True).T[:, :2].tolist()
                    final_coords = post_process_coords(corner_coords, imsize=cam_imsize[cam_key])
                    if final_coords is None:
                        continue

                    min_x, min_y, max_x, max_y = final_coords
                    all_2d_boxes.append([min_x, min_y, max_x, max_y, cam_index])

                    if not has_proj:
                        gt_boxes_2d.append([min_x, min_y, max_x, max_y, cam_index])
                        has_proj = True

                if not has_proj:
                    gt_boxes_2d.append([0.0, 0.0, 1.0, 1.0, -1])

            pc_dataset[pc_rel_path]["gt_boxes_2d"] = np.array(gt_boxes_2d, dtype=np.float32)
            pc_dataset[pc_rel_path]["all_2d_boxes"] = np.array(all_2d_boxes, dtype=np.float32)

            if vis_check:
                visualize_gt_boxes_2d(pc_dataset[pc_rel_path], draw_all=True)
    return pc_dataset



def main(args):
    all_pc_dataset = {}

    for prefix in args.data_prefix:
        rotate_point_cloud(args.data_root, args.rot_pc, prefix, args.sensor_type, workers=args.workers)

        pc_data = create_dataset_from_json(
            args.data_root, prefix, args.rot_pc, args.sensor_type,
            args.calib_dir, args.use_lidar, args.use_cam, args.vis_check
        )
        all_pc_dataset.update(pc_data)

    # train/val 划分逻辑
    train_pc_keys, val_pc_keys, dropped_tokens = split_dataset_divisible(all_pc_dataset, args.trainval_ratio, m=args.batch_size, seed=42, drop_from='val')
    print(f"Train={len(train_pc_keys)}, Val={len(val_pc_keys)}, DroppedTokens={len(dropped_tokens)}")



    with open(Path(args.data_root) / 'sunlakes_infos_train.pkl', 'wb') as f:
        pickle.dump([all_pc_dataset[k] for k in train_pc_keys], f)
    with open(Path(args.data_root) / 'sunlakes_infos_val.pkl', 'wb') as f:
        pickle.dump([all_pc_dataset[k] for k in val_pc_keys], f)
    if args.create_gt:
        create_groundtruth_database(args.data_root,args.with_cam_gt,args.vis_check)

    print(
        f"✅ Finished.\n🖼️ PC Train={len(train_pc_keys)}, Val={len(val_pc_keys)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nuscenes Converter Tool')
    parser.add_argument('--data_root', default='../data/sunlakes',
                        help='your data root for raw data')
    parser.add_argument('--data_prefix', nargs='+',
                        default=['2026-01-04_18_02_01-export'],
                        help='one or more sub paths for raw data')
    parser.add_argument('--rot_pc', type=float, default=-142.5,
                        help='rotation angle of raw point cloud')
    parser.add_argument('--trainval_ratio', type=float, default=0.8,
                        help='split ratio of train and val dataset')
    parser.add_argument('--sensor_type', choices=['os128', 'os64', 'rs16'], default='os64down',
                        help='选择点云标签目录：os128=128_128b_label（默认），os64=64_64b_label，rs16=16_16b_label')
    parser.add_argument(
        "--calib_dir",
        type=str,
        default=None,
        help="相机标定目录(默认使用 <data_root>/spatial_results)",
    )
    parser.add_argument('--batch_size', type=int, default=16, help='batch size when used in BEVfusion')
    parser.add_argument('--workers', type=int, default=min(20, os.cpu_count()),
                    help='并行进程数，受磁盘I/O限制）')
    parser.add_argument('--create_gt', action='store_true', default=True, help='create ground truth or not')
    parser.add_argument('--use_lidar', action='store_true', default=True, help='create lidar info or not')
    parser.add_argument('--use_cam', action='store_true', default=True, help='create cam info or not')
    parser.add_argument('--with_cam_gt', action='store_true', default=True, help='use camera gt database or not')
    parser.add_argument('--vis_check', action='store_true', default=False, help='visualization to check if pc2img and img_gt is correct')
    args = parser.parse_args()

    main(args)
