import os
import numpy as np
# import matplotlib.pyplot as plt
import colorsys
import random
import json
import torch
import point_cloud_utils as pcu
# from src.edge_extraction.extract_util import bezier_curve_length
from pathlib import Path
import re

# def visualize_pred_gt(all_pred_points, all_gt_points, name, save_fig=False, show_fig=True):
#     # all_pred_points = []

#     ax = plt.figure(dpi=120).add_subplot(projection='3d')

#     x = [k[0] for k in all_pred_points]
#     y = [k[1] for k in all_pred_points]
#     z = [k[2] for k in all_pred_points]
#     # print("max xyz:", max(x), max(y), max(z))
#     ax.scatter(x, y, z, c='r', marker='o', s=0.5, linewidth=1, alpha=1, cmap='spectral')
#     # ---------------------------------plot the gt---------------------------------
#     x = [k[0] for k in all_gt_points]
#     y = [k[1] for k in all_gt_points]
#     z = [k[2] for k in all_gt_points]
#     ax.scatter(x, y, z, c='g', marker='o', s=0.5, linewidth=1, alpha=1, cmap='spectral')

#     # ax.axis('auto')
#     plt.axis('off')
#     # plt.xlabel("X axis")
#     # plt.ylabel("Y axis")

#     ax.view_init(azim=60, elev=60)
#     range_size = [0, 1]
#     ax.set_zlim3d(range_size[0], range_size[1])
#     plt.axis([range_size[0], range_size[1], range_size[0], range_size[1]])
#     if save_fig:
#         plt.savefig(os.path.join(vis_dir, name + ".png"), bbox_inches='tight')
#     if show_fig:
#         plt.show()

def get_pred_points_from_obj(fn, sample_scale=2000):
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
        num = int(np.linalg.norm(l[0] - l[1]) // 0.001)
        # print(num)
        linspace = np.linspace(0, 1, num)
        sampled_points = np.concatenate(
            (linspace[:, None] * l[1] + (1-linspace)[:, None] * l[0], sampled_points), axis=0)
    # print(f"{fn.stem}, {sampled_points.shape}")
    x_max, y_max, z_max = np.max(sampled_points, axis=0)
    x_min, y_min, z_min = np.min(sampled_points, axis=0)
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    scale = 1 / max(x_range, y_range, z_range)
    poi_center = np.array(
        [((x_min + x_max) / 2), ((y_min + y_max) / 2), ((z_min + z_max) / 2)]) * scale
    set_location = [0.5, 0.5, 0.5] - poi_center
    sampled_points = sampled_points * scale + set_location
    return sampled_points.astype(np.float32), scale

# def get_pred_points_and_directions(
#     json_path,
#     sample_resolution=0.005,
# ):
#     with open(json_path, "r") as f:
#         json_data = json.load(f)

#     curve_paras = np.array(json_data["curves_ctl_pts"]).reshape(-1, 3)
#     curves_ctl_pts = curve_paras.reshape(-1, 4, 3)
#     lines_end_pts = np.array(json_data["lines_end_pts"]).reshape(-1, 2, 3)

#     num_curves = len(curves_ctl_pts)
#     num_lines = len(lines_end_pts)

#     all_curve_points = []
#     all_curve_directions = []

#     # # -----------------------------------for Cubic Bezier-----------------------------------
#     if num_curves > 0:
#         for i, each_curve in enumerate(curves_ctl_pts):
#             each_curve = np.array(each_curve).reshape(4, 3)  # shape: (4, 3)
#             sample_num = int(
#                 bezier_curve_length(each_curve, num_samples=100) // sample_resolution
#             )
#             t = np.linspace(0, 1, sample_num)
#             matrix_u = np.array([t**3, t**2, t, [1] * sample_num]).reshape(
#                 4, sample_num
#             )

#             matrix_middle = np.array(
#                 [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
#             )

#             matrix = np.matmul(
#                 np.matmul(matrix_u.T, matrix_middle), each_curve
#             ).reshape(sample_num, 3)

#             all_curve_points += matrix.tolist()

#             # Calculate the curve directions (derivatives)
#             derivative_u = 3 * t**2
#             derivative_v = 2 * t

#             # Derivative matrices for x, y, z
#             dx = (
#                 (
#                     -3 * each_curve[0][0]
#                     + 9 * each_curve[1][0]
#                     - 9 * each_curve[2][0]
#                     + 3 * each_curve[3][0]
#                 )
#                 * derivative_u
#                 + (6 * each_curve[0][0] - 12 * each_curve[1][0] + 6 * each_curve[2][0])
#                 * derivative_v
#                 + (-3 * each_curve[0][0] + 3 * each_curve[1][0])
#             )

#             dy = (
#                 (
#                     -3 * each_curve[0][1]
#                     + 9 * each_curve[1][1]
#                     - 9 * each_curve[2][1]
#                     + 3 * each_curve[3][1]
#                 )
#                 * derivative_u
#                 + (6 * each_curve[0][1] - 12 * each_curve[1][1] + 6 * each_curve[2][1])
#                 * derivative_v
#                 + (-3 * each_curve[0][1] + 3 * each_curve[1][1])
#             )

#             dz = (
#                 (
#                     -3 * each_curve[0][2]
#                     + 9 * each_curve[1][2]
#                     - 9 * each_curve[2][2]
#                     + 3 * each_curve[3][2]
#                 )
#                 * derivative_u
#                 + (6 * each_curve[0][2] - 12 * each_curve[1][2] + 6 * each_curve[2][2])
#                 * derivative_v
#                 + (-3 * each_curve[0][2] + 3 * each_curve[1][2])
#             )
#             for i in range(sample_num):
#                 direction = np.array([dx[i], dy[i], dz[i]])
#                 norm_direction = direction / np.linalg.norm(direction)
#                 all_curve_directions.append(norm_direction)

#     all_line_points = []
#     all_line_directions = []
#     # # -------------------------------------for Line-----------------------------------------
#     if num_lines > 0:
#         for i, each_line in enumerate(lines_end_pts):
#             each_line = np.array(each_line).reshape(2, 3)  # shape: (2, 3)
#             sample_num = int(
#                 np.linalg.norm(each_line[0] - each_line[-1]) // sample_resolution
#             )
#             t = np.linspace(0, 1, sample_num)

#             matrix_u_l = np.array([t, [1] * sample_num])
#             matrix_middle_l = np.array([[-1, 1], [1, 0]])

#             matrix_l = np.matmul(
#                 np.matmul(matrix_u_l.T, matrix_middle_l), each_line
#             ).reshape(sample_num, 3)
#             all_line_points += matrix_l.tolist()

#             # Calculate the direction vector for the line segment
#             direction = each_line[1] - each_line[0]
#             norm_direction = direction / (np.linalg.norm(direction) + 1e-6)

#             for point in matrix_l:
#                 all_line_directions.append(norm_direction)

#     all_curve_points = np.array(all_curve_points).reshape(-1, 3)
#     all_line_points = np.array(all_line_points).reshape(-1, 3)
#     sampled_points =  np.concatenate([all_curve_points, all_line_points], axis=0).reshape(-1, 3)
#     # # if "bed" in json_path:
#     # #     return sampled_points
#     # if "chair_0557" not in json_path:
#     #     return sampled_points
#     # if "chair" in json_path or "bed" in json_path or "monitor" in json_path:
#     #     x_max, y_max, z_max = np.max(sampled_points, axis=0)
#     #     x_min, y_min, z_min = np.min(sampled_points, axis=0)
#     #     x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
#     #     scale = 1 / max(x_range, y_range, z_range)
#     #     poi_center = np.array(
#     #         [((x_min + x_max) / 2), ((y_min + y_max) / 2), ((z_min + z_max) / 2)]) * scale
#     #     set_location = [0.5, 0.5, 0.5] - poi_center
#     #     sampled_points = sampled_points * scale + set_location
#     return sampled_points
    # return all_curve_points, all_line_points, all_curve_directions, all_line_directions

def sample_points_by_grid(pred_points, num_voxels_per_axis=64, min_bound=None, max_bound=None):
    bbox_size = np.array([1, 1, 1])
    # The size per-axis of a single voxel
    sizeof_voxel = bbox_size / num_voxels_per_axis

    # Use the existing function to downsample the point cloud based on voxel size
    pred_sampled = pcu.downsample_point_cloud_on_voxel_grid( sizeof_voxel, pred_points, min_bound=min_bound, max_bound=max_bound )

    return pred_sampled.astype(np.float32)

def get_pred_points(json_path, curve_type="cubic", sample_num=100):
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


def get_gt_points(name):
    base_dir = "D:\\workspace\\EMAP"
    objs_dir = os.path.join(base_dir, "obj")
    obj_names = os.listdir(objs_dir)
    obj_names.sort()
    index_obj_names = {}
    for obj_name in obj_names:
        index_obj_names[obj_name[:8]] = obj_name
    # print(index_obj_names)

    json_feats_path = os.path.join(base_dir, "chunk_0000_feats.json")
    with open(json_feats_path, 'r') as f:
        json_data_feats = json.load(f)
    json_stats_path = os.path.join(base_dir, "chunk_0000_stats.json")
    with open(json_stats_path, 'r') as f:
        json_data_stats = json.load(f)

    # get the normalize scale to help align the nerf points and gt points
    [x_min, y_min, z_min, x_max, y_max, z_max, x_range, y_range, z_range] = json_data_stats[name]["bbox"]
    scale = 1 / max(x_range, y_range, z_range)
    # print("normalize scale:", scale)
    poi_center = np.array([((x_min + x_max) / 2), ((y_min + y_max) / 2), ((z_min + z_max) / 2)]) * scale
    # print("poi:", poi_center)
    set_location = [0.5, 0.5, 0.5] - poi_center  # based on the rendering settings

    obj_path = os.path.join(objs_dir, index_obj_names[name])
    with open(obj_path, encoding='utf-8') as file:
        data = file.readlines()
    vertices_obj = [each.split(' ') for each in data if each.split(' ')[0] == 'v']
    vertices_xyz = [[float(v[1]), float(v[2]), float(v[3].replace('\n', ''))] for v in vertices_obj]

    edge_pts = []
    edge_pts_raw = []
    for each_curve in json_data_feats[name]:
        if each_curve['sharp']:
            each_edge_pts = [vertices_xyz[i] for i in each_curve['vert_indices']]
            edge_pts_raw += each_edge_pts

            gt_sampling = []
            each_edge_pts = np.array(each_edge_pts)
            for index in range(len(each_edge_pts) - 1):
                next = each_edge_pts[index + 1]
                current = each_edge_pts[index]
                num = int(np.linalg.norm(next - current) // 0.01)
                linspace = np.linspace(0, 1, num)
                gt_sampling.append(linspace[:, None] * current + (1 - linspace)[:, None] * next)
            each_edge_pts = np.concatenate(gt_sampling).tolist()
            edge_pts += each_edge_pts


    edge_pts_raw = np.array(edge_pts_raw) * scale + set_location
    edge_pts = np.array(edge_pts) * scale + set_location

    return edge_pts_raw.astype(np.float32), edge_pts.astype(np.float32)


def compute_chamfer_distance(pred_sampled, gt_points, metrics):
    chamfer_dist = pcu.chamfer_distance(pred_sampled, gt_points)
    metrics["chamfer"].append(chamfer_dist)
    print("chamfer_dist:", chamfer_dist)
    return metrics


def compute_precision_recall_IOU(pred_sampled, gt_points, metrics, thresh=0.02):
    dists_a_to_b, _ = pcu.k_nearest_neighbors(pred_sampled, gt_points,
                                              k=1)  # k closest points (in pts_b) for each point in pts_a
    correct_pred = np.sum(dists_a_to_b < thresh)
    precision = correct_pred / len(dists_a_to_b)
    metrics["precision"].append(precision)

    dists_b_to_a, _ = pcu.k_nearest_neighbors(gt_points, pred_sampled, k=1)
    correct_gt = np.sum(dists_b_to_a < thresh)
    recall = correct_gt / len(dists_b_to_a)
    metrics["recall"].append(recall)

    fscore = 2 * precision * recall / (precision + recall)
    metrics["fscore"].append(fscore)

    intersection = min(correct_pred, correct_gt)
    union = len(dists_a_to_b) + len(dists_b_to_a) - max(correct_pred, correct_gt)

    IOU = intersection / union
    metrics["IOU"].append(IOU)
    print("precision:", precision, "recall:", recall, "fscore:", fscore, "IOU:", IOU)
    return metrics



# objs = ['00004733', '00007042']

# for obj in objs:
# points = get_pred_points_and_directions(f"model_curve\\chair_0557_emap.json", sample_resolution=0.001)
# np.savetxt(f"chair_0557_pred_points.xyz", points, delimiter=" ")
if __name__ == "__main__":
    save_curve_dir = "scale_1d_res"
    vis_dir = "./visualization"
    os.makedirs(vis_dir, exist_ok=True)

# part = sorted([i.name for i in Path("ABC_curve_old").iterdir()])

    result_names = [each for each in Path(save_curve_dir).rglob("*_stage2_cubic.json")]

    # for i in full:
    #     if i not in result_names:
    #         print(i)
    result_names.sort()
    print(len(result_names))

    metrics = {
            "chamfer": [],
            "precision": [],
            "recall": [],
            "fscore": [],
            "IOU": []
        }

    test_only_line = False
    if test_only_line:
        with open("only_line_list.txt", 'r') as f:
            line_obj_names = f.readlines()
        line_obj_names = [each.replace('\n', '') for each in line_obj_names]
        print(line_obj_names)
        print("number of objs containing only lines:", len(line_obj_names))

    for i, result_name in enumerate(result_names):       # result_name like: record_00000006_0.7_stage2_cubic.json
        # if result_name in part:
        #     continue
        name = result_name.name.split('_')[1]    # name like: 00000006
        # print(result_name)
        # name = '_'.join(result_name.split('_')[:2])
        # print(name)
        if test_only_line and (name not in line_obj_names):
            continue
        print("-" * 50)
        print("processing:", i, ", name:", name)

        # result_path = os.path.join(save_curve_dir, result_name)
        result_path = result_name
        # result_path = f"gaussian_res\\{name}\\version2\\record_{name}_stage2_cubic.obj"
        gt_points_raw, gt_points = get_gt_points(name)
        # gt_points_raw, scale = get_pred_points_from_obj(f"ModelNet_GT_edge\\{name}_edge_GT.obj")
        gt_points_raw = sample_points_by_grid(gt_points_raw)

        pred_points = get_pred_points(result_path, curve_type="cubic", sample_num=500)
        # pred_points = get_pred_points_and_directions(result_path, sample_resolution=0.002)
        # poi_center = np.mean(pred_points, axis=0)
        # set_location = [0.5, 0.5, 0.5] - poi_center
        # pred_points = pred_points * scale + set_location
        # pred_points = get_pred_points_from_obj(result_path, sample_scale=500)
        pred_sampled = sample_points_by_grid(pred_points)

        # np.savetxt(f"Replica_emap\\{name}_emap.xyz", pred_points, delimiter=" ")
        # print(f"Replica_emap\\{name}_emap.xyz saved")
        # np.savetxt(f"ABC_pred\\{name}_gt_point_raw.xyz", gt_points_raw, delimiter=" ")
        metrics = compute_chamfer_distance(pred_sampled, gt_points_raw, metrics)
        metrics = compute_precision_recall_IOU(pred_sampled, gt_points_raw, metrics, thresh=0.02)
        print("raw preds:", pred_points.shape, ", sampled preds:", pred_sampled.shape, ", gt_raw shape:", gt_points_raw.shape, ", gt shape:", gt_points.shape)
        # visualize_pred_gt(pred_points, gt_points, name, save_fig=False, show_fig=True)
    for key, value in metrics.items():
        metrics[key] = round(np.mean(value), 4)
    print("total CADs:", len(result_names))
    print(metrics)