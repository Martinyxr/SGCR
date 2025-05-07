import json
import numpy as np
import re
from eval import compute_chamfer_distance, compute_precision_recall_IOU, get_gt_points, sample_points_by_grid

def get_pred_points(fn, sample_scale=2000):
    with open(fn, 'r') as f:
        data = f.readlines()
    points = [re.sub(" +", " ", each).rstrip().split(' ')[1:]
              for each in data if each.split(' ')[0] == 'v']
    lines = [re.sub(" +", " ", each).rstrip().split(' ')[1:]
             for each in data if each.split(' ')[0] == 'l']
    points = [[float(p[0]), float(p[1]), float(p[2])]
              for p in points]
    lines = [[int(l[0]), int(l[1])] for l in lines]
    points = np.array(points, dtype=np.float32)
    lines = np.array(lines, dtype=np.int32) - 1
    lines = points[lines]
    lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
    sampled_points = np.zeros((0, 3), dtype=np.float32)
    for i, l in enumerate(lines):
        num = int(np.linalg.norm(l[0] - l[1]) // 0.001) + 1
        # print(num)
        linspace = np.linspace(0, 1, num)
        sampled_points = np.concatenate(
            (linspace[:, None] * l[1] + (1-linspace)[:, None] * l[0], sampled_points), axis=0)
    # print(f"{fn.stem}, {sampled_points.shape}")
    # x_max, y_max, z_max = np.max(sampled_points, axis=0)
    # x_min, y_min, z_min = np.min(sampled_points, axis=0)
    # x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    # scale = 1 / max(x_range, y_range, z_range)
    # poi_center = np.array(
    #     [((x_min + x_max) / 2), ((y_min + y_max) / 2), ((z_min + z_max) / 2)]) * scale
    # set_location = [0.5, 0.5, 0.5] - poi_center
    # sampled_points = sampled_points * scale + set_location
    return sampled_points.astype(np.float32)

def get_pred_points_json(json_path, curve_type="cubic", sample_num=100):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    curves_ctl_pts = json_data['curves_ctl_pts']
    num_curves = len(curves_ctl_pts)
    print("num_curves:", num_curves)

    t = np.linspace(0, 1, sample_num)
    if curve_type == "cubic":
        # # -----------------------------------for Cubic Bezier-----------------------------------
        matrix_u = np.array([t ** 3, t ** 2, t, 1], dtype=object)
        matrix_middle = np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ])
    elif curve_type == "line":
        # # -------------------------------------for Line-----------------------------------------
        matrix_u = np.array([t, 1], dtype=object)
        matrix_middle = np.array([
            [-1, 1],
            [1, 0]
        ])

    all_points = []
    for i, each_curve in enumerate(curves_ctl_pts):
        each_curve = np.array(each_curve)  # shape: (4, 3)
        # each_curve = (each_curve / 256 * 2.4) - 1.2   # based on the settings of extract point cloud

        matrix = np.matmul(np.matmul(matrix_u.T, matrix_middle), each_curve)
        for i in range(sample_num):
            all_points.append([matrix[0][i], matrix[1][i], matrix[2][i]])

    # exchange X and Y axis, do not know why yet... ...
    all_points = np.array([[pts[0], pts[1], pts[2]] for pts in all_points])
    return np.array(all_points)

if __name__ == "__main__":

    json_path = "D:\\workspace\\gaussian-splatting_NEF\\parametric_curve\\scale_1d_res\\00003014_wo_opa\\version2\\record_00003014_wo_opa_stage2_cubic.obj"
    pred_points = get_pred_points(json_path)
    np.savetxt("00003014_wo_opa_stage_2.xyz", pred_points)
    # metrics = {
    #     "chamfer": [],
    #     "precision": [],
    #     "recall": [],
    #     "fscore": [],
    #     "IOU": []
    # }
    # json_path = "D:\\workspace\\gaussian-splatting_NEF\\parametric_curve\\scale_1d_res\\2030_muge2_1d\\version2\\record_2030_muge2_1d_stage2_cubic.obj"
    # pred_points = get_pred_points(json_path)
    # pred_sampled = sample_points_by_grid(pred_points, num_voxels_per_axis=128)
    # gt_points_raw, gt_points = get_gt_points("00002030")
    # # gt_points_raw, scale = get_pred_points_from_obj(f"ModelNet_GT_edge\\{name}_edge_GT.obj")
    # gt_points_raw = sample_points_by_grid(gt_points_raw, num_voxels_per_axis=128)
    # metrics = compute_chamfer_distance(pred_sampled, gt_points_raw, metrics)
    # metrics = compute_precision_recall_IOU(pred_sampled, gt_points_raw, metrics, thresh=0.008)
    # for key, value in metrics.items():
    #     metrics[key] = round(np.mean(value), 4)
    # print(metrics)

