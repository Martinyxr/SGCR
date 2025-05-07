# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import time
import json
import point_cloud_utils as pcu
from tqdm import tqdm
from glob import glob
import argparse

from ChamferDistancePytorch import fscore
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D

# https://github.com/ThibaultGROUEIX/ChamferDistancePytorch

C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def save_edges_in_obj_format(filename, v_list, e_list):
    with open(filename, "w") as f:
        for vertex in v_list:
            if len(vertex) == 3:
                f.write('v ' + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]))
            else:
                f.write('v ' + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]))
                f.write(' ' + str(vertex[3]) + " " + str(vertex[4]) + " " + str(vertex[5]))
            f.write('\n')

        f.write("\n")

        for edge in e_list:
            f.write('l ' + str(edge[0] + 1) + " " + str(edge[1] + 1))
            f.write('\n')


def get_matrix_t_numpy(curve_type="cubic", num=100):
    matrix_t = []
    if curve_type == "cubic":
        for t in np.linspace(0, 1, num):
            each_matrix_t = np.array([
                t * t * t,
                t * t,
                t,
                1
            ])
            matrix_t.append(each_matrix_t)
    elif curve_type == "line":
        for t in np.linspace(0, 1, num):
            each_matrix_t = np.array([
                t,
                1
            ])
            matrix_t.append(each_matrix_t)

    matrix_t = np.stack(matrix_t, axis=0).astype(float)
    return matrix_t


class Curves_Model(nn.Module):
    def __init__(self, n_curves=12, initial_params=None, initial_scale=None, curve_type="line", sample_num=100):
        super(Curves_Model, self).__init__()
        self.curve_type = curve_type
        self.n_curves = n_curves
        self.weight = torch.ones((n_curves, 4, 1)).cuda()  # [n_curves, 4, 3]

        if self.curve_type == "line":
            self.n_ctl_points = 2
            self.matrix_w = torch.tensor([
                [-1, 1],
                [1, 0]
            ]).float().cuda()
        elif self.curve_type == "cubic":
            self.n_ctl_points = 4
            self.matrix_w = torch.tensor([
                [-1, 3, -3, 1],
                [3, -6, 3, 0],
                [-3, 3, 0, 0],
                [1, 0, 0, 0]
            ]).float().cuda()
            self.weight = nn.Parameter(self.weight)

        self.matrix_t = self.get_matrix_t(num=sample_num)  # [sample_num, 2/4]

        if initial_params is None:
            params = torch.rand(n_curves, self.n_ctl_points, 3, requires_grad=True).cuda()  # [n_curves, 2/4, 3]
        else:
            params = initial_params.cuda()
        assert params.shape == (n_curves, self.n_ctl_points, 3)
        self.params = nn.Parameter(params)

    def initialize_params(self, pts_target, pts_opacity, init_mode):  # pts_target: # [1, N, 3]     pts_opacity: numpy [N]

        if init_mode == "center":
            center_pts = torch.mean(pts_target.squeeze(), axis=0)
            self.params.requires_grad = False
            for i in range(len(self.params)):
                for j in range(len(self.params[i])):
                    self.params[i][j] = center_pts
            self.params.requires_grad = True
        elif init_mode == "boundingbox_random":
            x_min = torch.min(pts_target.squeeze()[:, 0])
            x_max = torch.max(pts_target.squeeze()[:, 0])
            y_min = torch.min(pts_target.squeeze()[:, 1])
            y_max = torch.max(pts_target.squeeze()[:, 1])
            z_min = torch.min(pts_target.squeeze()[:, 2])
            z_max = torch.max(pts_target.squeeze()[:, 2])

            self.params.requires_grad = False
            param_x = torch.rand(self.n_curves, self.n_ctl_points, 1).cuda() * (
                    x_max - x_min) + x_min  # [n_curves, 2/4, 1]
            param_y = torch.rand(self.n_curves, self.n_ctl_points, 1).cuda() * (y_max - y_min) + y_min
            param_z = torch.rand(self.n_curves, self.n_ctl_points, 1).cuda() * (z_max - z_min) + z_min
            params = torch.cat([param_x, param_y, param_z], dim=-1)
            self.params = nn.Parameter(params)
            self.params.requires_grad = True
        elif init_mode == "pts_random":
            idx = torch.randint(0, pts_target.shape[1] - 1, size=(self.n_curves, self.n_ctl_points)).cuda()
            idx = idx.view(-1, 1).squeeze()  # [N]
            params = pts_target[:, idx, :]
            params = params.view(self.n_curves, self.n_ctl_points, 3)
            # print("params:", params.shape)
            self.params.requires_grad = False
            self.params = nn.Parameter(params)
            self.params.requires_grad = True
        elif init_mode == "pts_near":
            knn = 2
            np_pts_target = pts_target.squeeze(0).cpu().detach().numpy()
            dists_a_to_a, corrs_a_to_a = pcu.k_nearest_neighbors(np_pts_target, np_pts_target, k=knn)

            idx_0 = np.random.randint(pts_target.shape[1])

            idx_1 = corrs_a_to_a[idx_0][1]

            params_0 = pts_target[:, idx_0]
            params_1 = pts_target[:, idx_1]
            params = torch.cat([params_0, params_1], dim=0).unsqueeze(0).repeat(self.n_curves, 1, 1)    # [1, 2, 3]
            if self.n_ctl_points == 4:
                params = self.torch_Line2Cubic(params)

            self.params.requires_grad = False
            self.params = nn.Parameter(params)
            self.params.requires_grad = True
        elif init_mode == "max_opacity":
            idx_0 = np.argmax(pts_opacity)

            params_0 = pts_target[:, idx_0]

            np_pts_target = pts_target.squeeze(0).cpu().detach().numpy()    # [N, 3]

            dists_a_to_a, corrs_a_to_a = pcu.k_nearest_neighbors(np_pts_target, np_pts_target, k=2)

            idx_1 = corrs_a_to_a[idx_0][1]

            params_1 = pts_target[:, idx_1]
            params = torch.cat([params_0, params_1], dim=0).unsqueeze(0).repeat(self.n_curves, 1, 1)  # [1, 2, 3]
            if self.n_ctl_points == 4:
                params = self.torch_Line2Cubic(params)
            self.params.requires_grad = False
            self.params = nn.Parameter(params)
            self.params.requires_grad = True

    def get_matrix_t(self, num):
        matrix_t = []
        if self.curve_type == "line":
            for t in np.linspace(0, 1, num):
                each_matrix_t = torch.tensor([
                    t,
                    1
                ])
                matrix_t.append(each_matrix_t)
        elif self.curve_type == "cubic":
            for t in np.linspace(0, 1, num):
                each_matrix_t = torch.tensor([
                    t * t * t,
                    t * t,
                    t,
                    1
                ])
                matrix_t.append(each_matrix_t)

        matrix_t = torch.stack(matrix_t, axis=0).float().cuda()
        return matrix_t

    def torch_Line2Cubic(self, params):     # [1, 2, 3]
        curves_ctl_pts = params.detach().cpu().numpy()

        curves_ctl_pts_new = []
        for each_curve in curves_ctl_pts:
            each_curve = np.array(each_curve)
            extra_pts1 = 2 / 3 * each_curve[0] + 1 / 3 * each_curve[1]
            extra_pts2 = 1 / 3 * each_curve[0] + 2 / 3 * each_curve[1]
            new_curve = np.array([each_curve[0], extra_pts1, extra_pts2, each_curve[1]]).tolist()
            curves_ctl_pts_new.append(new_curve)
        curves_ctl_pts_new = np.array(curves_ctl_pts_new, dtype=np.float32)
        curves_ctl_pts_new = torch.from_numpy(curves_ctl_pts_new).cuda()

        return curves_ctl_pts_new

    def forward(self, use_weight=False):
        matrix1 = torch.einsum('ik,kj->ij',
                               [self.matrix_t, self.matrix_w])  # shape: [sample_num, 4] * [4, 4] = [sample_num, 4]
        if self.curve_type == "line" or not use_weight:

            matrix2 = torch.einsum('ik,nkj->nij', [matrix1, self.params])  # shape: [sample_num, 4] * [n_curves, 4, 3] = [n_curves, sample_num, 3]
        else:
            matrix_weight = torch.einsum('ik,nkj->nij', [matrix1, self.weight])  # shape: [sample_num, 4] * [n_curves, 4, 1] = [n_curves, sample_num, 1]
            matrix2 = torch.einsum('ik,nkj->nij', [matrix1,
                                                   self.params * self.weight])  # shape: [sample_num, 4] * [n_curves, 4, 3] = [n_curves, sample_num, 3]
            matrix2 = matrix2 / matrix_weight  # [n_curves, sample_num, 3] / [n_curves, sample_num, 1]

        pts_curve = matrix2.reshape(1, -1, 3)  # shape: [1, n_curves * sample_num, 3]

        multiply = 5  # default 5
        pts_curve_m = pts_curve.repeat(1, multiply, 1)  # shape: [1, n_curves * sample_num * multiply, 3]
        noise = torch.randn_like(pts_curve_m)
        noise = radius * noise
        pts_curve_m = pts_curve_m + noise
        return pts_curve, pts_curve_m, self.params, self.weight


def optimize_one_curve(max_iters, pts_target, pts_opacity, alpha=1, curve_type="line", repeat_times=0):
    global lr
    chamLoss = dist_chamfer_3D.chamfer_3DDist()
    curve_model = Curves_Model(n_curves=1, curve_type=curve_type)

    curve_model.initialize_params(pts_target, pts_opacity, init_mode="pts_near")

    optimizer = torch.optim.Adam(curve_model.parameters(), lr=lr)

    for iters in range(max_iters):
        pts_curve, pts_curve_m, current_params, current_weight = curve_model(use_weight=use_rational_bezier)

        dist1, dist2, idx1, idx2 = chamLoss(pts_curve_m, pts_target)  # [1, 500]  [1, N]   [1, 500]   [1, N]

        near_idx = dist2 < dis_threshold_1  # dis_threshold_1
        pts_near = pts_target[near_idx].unsqueeze(0)
        if pts_near.shape[1] == 0:
            break

        dist1, dist2, idx1, idx2 = chamLoss(pts_curve_m, pts_near)

        chamfer_loss_1 = alpha * torch.sqrt(dist1).mean()
        chamfer_loss_2 = torch.sqrt(dist2).mean()

        loss = chamfer_loss_1 + chamfer_loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    score = torch.sqrt(dist1).mean().detach().cpu().numpy()

    return current_params, current_weight, pts_curve, pts_curve_m, score


def update_pts_target(pts_curve, pts_target, pts_opacity, repeat_times):
    pts_curve = pts_curve.squeeze(0).cpu().detach().numpy()
    pts_target = pts_target.squeeze(0).cpu().detach().numpy()

    length = np.linalg.norm(pts_curve[-1] - pts_curve[0])

    if pts_target.shape[0] < curve_min_points:
        return pts_target, pts_opacity, 0

    dists_a_to_b_1, corrs_a_to_b_1 = pcu.k_nearest_neighbors(pts_curve, pts_target, k=1)

    risk_idx = corrs_a_to_b_1[dists_a_to_b_1 > 0.02]


    dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(pts_curve, pts_target, k=100)  # [100, k]

    distance = dis_threshold_1   # dis_threshold_1
    interval = length / (pts_curve.shape[0] - 1)
    mask_end_num = int(dis_threshold_1 / interval) - 1


    delete_index = corrs_a_to_b[dists_a_to_b < distance]
    delete_index = list(set(delete_index))

    pts_target = np.delete(pts_target, delete_index, axis=0)
    pts_opacity = np.delete(pts_opacity, delete_index, axis=0)

    if pts_target.shape[0] == 0:
        return pts_target, pts_opacity, len(delete_index)

    dists_b_to_b, corrs_b_to_b = pcu.k_nearest_neighbors(pts_target, pts_target, k=curve_min_points - 1)
    dists_b_to_b = np.max(dists_b_to_b, axis=-1)

    delete_isolate = dists_b_to_b > 0.1

    if pts_target.shape[0] > 1:
        pts_target = np.delete(pts_target, delete_isolate, axis=0)
        pts_opacity = np.delete(pts_opacity, delete_isolate, axis=0)

    return pts_target, pts_opacity, len(delete_index)


def Line2Cubic(curves_ctl_pts):
    curves_ctl_pts_new = []
    for each_curve in curves_ctl_pts:
        each_curve = np.array(each_curve)
        extra_pts1 = 2 / 3 * each_curve[0] + 1 / 3 * each_curve[1]
        extra_pts2 = 1 / 3 * each_curve[0] + 2 / 3 * each_curve[1]
        new_curve = np.array([each_curve[0], extra_pts1, extra_pts2, each_curve[1]]).tolist()
        curves_ctl_pts_new.append(new_curve)
    return curves_ctl_pts_new


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def sample_points_by_grid(pred_points, num_voxels_per_axis=64):
    bbox_size = np.array([1, 1, 1])
    sizeof_voxel = bbox_size / num_voxels_per_axis
    pred_sampled = pcu.downsample_point_cloud_on_voxel_grid(sizeof_voxel, pred_points)
    pred_sampled = pred_sampled.astype(np.float32)
    return pred_sampled


def get_pred_points(json_path, curve_type="cubic", sample_num=100):
    with open(json_path, "r") as f:
        json_data = json.load(f)

    curves_ctl_pts = json_data['curves_ctl_pts']
    curves_ctl_weights = json_data['curves_ctl_weights']
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
    else:
        raise NotImplementedError

    # matrix_u: matrix_t        matrix_middle: matrix_w
    all_points = []
    for i in range(len(curves_ctl_pts)):
        each_curve = np.array(curves_ctl_pts[i])
        each_curve_weight = np.array(curves_ctl_weights[i])
        each_curve_weight = np.tile(each_curve_weight, (1, 3))


        if curve_type == "line":
            matrix1 = np.matmul(matrix_u.T, matrix_middle)
            matrix2 = np.matmul(matrix1, each_curve)
        else:
            matrix1 = np.matmul(matrix_u.T, matrix_middle)
            matrix_weight = np.matmul(matrix1, each_curve_weight)
            matrix2 = np.matmul(matrix1, each_curve * each_curve_weight)
            matrix2 = matrix2 / matrix_weight

        for i in range(sample_num):
            all_points.append([matrix2[0][i], matrix2[1][i], matrix2[2][i]])

    return np.array(all_points)


def get_gt_points(name):
    name = name[:8]
    base_dir = "./gt_information"
    objs_dir = os.path.join(base_dir, "obj")
    obj_names = os.listdir(objs_dir)
    obj_names.sort()
    index_obj_names = {}
    for obj_name in obj_names:
        index_obj_names[obj_name[:8]] = obj_name

    json_feats_path = os.path.join(base_dir, "chunk_0000_feats.json")
    with open(json_feats_path, 'r') as f:
        json_data_feats = json.load(f)
    json_stats_path = os.path.join(base_dir, "chunk_0000_stats.json")
    with open(json_stats_path, 'r') as f:
        json_data_stats = json.load(f)

    # get the normalize scale to help align the nerf points and gt points
    [x_min, y_min, z_min, x_max, y_max, z_max, x_range, y_range, z_range] = json_data_stats[name]["bbox"]
    scale = 1 / max(x_range, y_range, z_range)
    poi_center = np.array([((x_min + x_max) / 2), ((y_min + y_max) / 2), ((z_min + z_max) / 2)]) * scale
    set_location = [0.5, 0.5, 0.5] - poi_center  # based on the rendering settings

    obj_path = os.path.join(objs_dir, index_obj_names[name])
    with open(obj_path, encoding='utf-8') as file:
        data = file.readlines()
    vertices_obj = [each.split(' ') for each in data if each.split(' ')[0] == 'v']
    vertices_xyz = [[float(v[1]), float(v[2]), float(v[3].replace('\n', ''))] for v in vertices_obj]

    edge_pts = []
    edge_pts_raw = []
    for each_curve in json_data_feats[name]:
        if each_curve["sharp"]:
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
    # print("chamfer_dist:", chamfer_dist)
    return metrics


def compute_precision_recall_IOU(pred_sampled, gt_points, metrics, thresh=0.02):
    dists_a_to_b, _ = pcu.k_nearest_neighbors(pred_sampled, gt_points,
                                              k=1)  
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
    return metrics


def eval_curve_metrics(object_id, save_curve_dir):
    metrics = {
        "chamfer": [],
        "precision": [],
        "recall": [],
        "fscore": [],
        "IOU": []
    }


    result_name = save_curve_dir + f"record_{object_id}_stage2_cubic.json"

    pred_points = get_pred_points(result_name, curve_type="cubic", sample_num=500)
    pred_sampled = sample_points_by_grid(pred_points)
    print("pred_points:", pred_points.shape, "pred_sampled:", pred_sampled.shape)

    np.savetxt(save_curve_dir + "pred_points.xyz", pred_points)

    gt_points_raw, gt_points = get_gt_points(object_id)
    print("gt_points:", gt_points.shape, "gt_points_raw:", gt_points_raw.shape)
    np.savetxt(save_curve_dir + "gt_points.xyz", gt_points)

    metrics = compute_chamfer_distance(pred_sampled, gt_points_raw, metrics)
    metrics = compute_precision_recall_IOU(pred_sampled, gt_points_raw, metrics, thresh=0.02)

    for key, value in metrics.items():
        metrics[key] = round(np.mean(value), 4)

    print(metrics)


def save_curve_from_json_to_obj(save_curve_dir, file_name, curve_type="line"):
    json_path = os.path.join(save_curve_dir, file_name)
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    curves_ctl_pts = json_data['curves_ctl_pts']
    curves_ctl_pts = np.array(curves_ctl_pts)
    curves_ctl_weights = json_data['curves_ctl_weights']
    curves_ctl_weights = np.array(curves_ctl_weights)
    print("min curves_ctl_weights:", np.mean(curves_ctl_weights))

    if curve_type == "line":
        matrix_w = np.array([
            [-1, 1],
            [1, 0]
        ]).astype(float)
    else:
        matrix_w = np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ]).astype(float)
    matrix_t = get_matrix_t_numpy(curve_type=curve_type, num=100)

    if curve_type == "line":
        matrix_t = get_matrix_t_numpy(curve_type=curve_type, num=100)
        matrix1 = np.einsum('ik,kj->ij', matrix_t, matrix_w)  # shape: [100, 4] * [4, 4] = [100, 4]
        matrix2 = np.einsum('ik,nkj->nij', matrix1, curves_ctl_pts)  # shape: [100, 4] * [n, 4, 3] = [n, 100, 3]
    else:
        curves_ctl_weights = np.tile(curves_ctl_weights, (1, 1, 3))
        matrix1 = np.einsum('ik,kj->ij', matrix_t, matrix_w)  # shape: [100, 4] * [4, 4] = [100, 4]

        matrix_weight = np.einsum('ik,nkj->nij', matrix1, curves_ctl_weights) # shape: [sample_num, 4] * [n_curves, 4, 1] = [n_curves, sample_num, 1]

        matrix2 = np.einsum('ik,nkj->nij', matrix1, curves_ctl_pts * curves_ctl_weights)  # shape: shape: [sample_num, 4] * [n_curves, 4, 3] = [n_curves, sample_num, 3]

        matrix2 = matrix2 / matrix_weight  # [n_curves, sample_num, 3] / [n_curves, sample_num, 1]
    pts_curve = matrix2

    obj_v, obj_l = [], []
    edge_count = 0
    for i in range(pts_curve.shape[0]):
        obj_v.extend(pts_curve[i])
        for j in range(len(pts_curve[i]) - 1):
            obj_l.append([edge_count + j, edge_count + j + 1])
        edge_count += len(pts_curve[i])

    obj_v, obj_l = np.array(obj_v), np.array(obj_l)

    save_edges_in_obj_format(save_curve_dir + f"/{file_name[:-5]}.obj", obj_v, obj_l)


if __name__ == "__main__":
    # Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", type=str, default="00000006")
    parser.add_argument("--gaussian_dir", type=str, default="../output/Gaussian/")
    parser.add_argument("--save_curve_dir", type=str, default="../output/Curve/")
    args = parser.parse_args()
    object_id = args.object_id
    save_obj_format = True

    # Hyper Parameters
    stage1 = True
    stage2 = True

    curve_min_points = 5   
    max_iters = 50         
    epoch_stage1 = 4000
    epoch_stage2 = 2000
    lr = 0.01               
    alpha = 2
    dis_threshold_1 = 0.02 
    score_threshold = 0.005
    max_search_times = 10        
    radius = 0.005
    gaussian_sample = 1
    use_rational_bezier = True
    eval_metrics = False

    dis_threshold_2 = 0.01
    lr2 = 0.05
    loss_end_pts_weight = 0.005  # 0.005
    use_opacity = True
    use_radius = False


    # chamfer_example
    print("->Loading Gaussians")
    gaussian_dir = args.gaussian_dir

    save_curve_dir = os.path.join(args.save_curve_dir, object_id)
    os.makedirs(save_curve_dir, exist_ok=True)


    print("-" * 50)
    pcd_path = os.path.join(gaussian_dir, object_id, "point_cloud", "iteration_6000", "point_cloud.ply")
    print("processing:", pcd_path)
    scene_name = f"{object_id}"

    plydata = PlyData.read(pcd_path)

    centers = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])
    color = np.asarray(plydata.elements[0]["f_dc_0"])

    print("xyz:", centers.shape, "opacities:", opacities.shape, "color:", color.shape)  # [N, 3]   [N]   [N]
    opacities = sigmoid(opacities)
    color = np.clip(C0 * color + 0.5, 0.0, 1.0)
    print("mean opacity:", opacities.mean())

    init_centers_target = torch.from_numpy(centers).float().unsqueeze(0).cuda()  # [1, N, 3]

    print("initial pts_target.shape:", init_centers_target.shape)
    curve_type = "line"

    # # stage 1: ------------------------per curve optimization-------------------------
    if stage1:
        print("Ready to conduct stage 1 ... ...")
        start = time.perf_counter()
        pts_target = init_centers_target.clone()
        pts_opacity = opacities.copy()

        if gaussian_sample > 1:
            gaussian_sample_pts = pts_target.repeat(1, gaussian_sample - 1, 1)
            noise = torch.randn_like(gaussian_sample_pts)
            noise = radius * noise
            gaussian_sample_pts = gaussian_sample_pts + noise
            pts_target = torch.cat([pts_target, gaussian_sample_pts], dim=1)

        cur_curves, cur_weights = [], []
        progress_bar = tqdm(range(epoch_stage1), desc="Stage1 progress")

        repeat_times, search_times = 0, 0
        min_score, cache = 100, {}
        for i in range(epoch_stage1):
            progress_bar.set_postfix({"Remain points:": f"{pts_target.shape[1]}", "repeat_times:": repeat_times})
            progress_bar.update(1)

            start_num = pts_target.shape[1]  # (1, N, 3)
            curves_params, curves_weight, pts_curve, pts_curve_m, score = optimize_one_curve(max_iters=max_iters,
                                                                                             pts_target=pts_target,
                                                                                             pts_opacity=pts_opacity,
                                                                                             alpha=alpha,
                                                                                             curve_type=curve_type,
                                                                                             repeat_times=repeat_times)
            if score < min_score:
                min_score = score
                cache["curves_params"] = curves_params
                cache["curves_weight"] = curves_weight
                cache["pts_curve"] = pts_curve
                cache["pts_curve_m"] = pts_curve_m

            if score > score_threshold:
                search_times += 1
                if search_times > max_search_times:
                    curves_params = cache["curves_params"]
                    curves_weight = cache["curves_weight"]
                    pts_curve = cache["pts_curve"]
                    pts_curve_m = cache["pts_curve_m"]
                    pts_target, pts_opacity, delete_num = update_pts_target(pts_curve, pts_target, pts_opacity, repeat_times)  # (N, 3)
                    if delete_num > 0:
                        cur_curves.append(np.array(curves_params.detach().cpu()))
                        cur_weights.append(np.array(curves_weight.detach().cpu()))
                    score = min_score
                else:
                    continue
            else:
                pts_target, pts_opacity, delete_num = update_pts_target(pts_curve, pts_target, pts_opacity, repeat_times)  # (N, 3)
                if delete_num > 0:
                    cur_curves.append(np.array(curves_params.detach().cpu()))
                    cur_weights.append(np.array(curves_weight.detach().cpu()))

            # print("cur_score:", score)
            search_times = 0
            min_score, cache = 100, {}
            pts_target = torch.from_numpy(pts_target).float().unsqueeze(0).cuda()

            if pts_target.shape[1] == start_num:
                repeat_times += 1
            else:
                repeat_times = 0
                min_score, cache = 100, {}

            if i == epoch_stage1 - 1:
                progress_bar.close()
            if pts_target.shape[1] <= 5 or repeat_times > 50:
                progress_bar.close()
                print("Stage1 finish epoch:", i)
                break

        progress_bar.close()

        cur_curves = np.array(cur_curves).squeeze(1)  # (total_curves, 1, 2/4, 3) to (total_curves, 2/4, 3)
        cur_weights = np.array(cur_weights).squeeze(1)
        print("remaining pts_target:", pts_target.shape)
        print("total curves:", cur_curves.shape)
        print("Total time comsumed", time.perf_counter() - start)

        json_data = {
            "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "scene_name": scene_name,
            'curves_ctl_pts': cur_curves.tolist(),
            'curves_ctl_weights': cur_weights.tolist(),
        }
        file_name = "record_" + scene_name + "_stage1_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        print("json file saved in", json_path)

        if save_obj_format:
            save_curve_from_json_to_obj(save_curve_dir, file_name, curve_type=curve_type)

    # stage 2:------------------------all curve refinement-------------------------
    if stage2:
        print("Ready to conduct stage 2 ... ...----------------------------------------------")
        print("opacity  max:", max(opacities), " min:", min(opacities), " mean:", np.mean(opacities))
        print("color  max:", max(color), " min:", min(color), " mean:", np.mean(color))
        init_opacities = torch.from_numpy(opacities).unsqueeze(0).cuda()
        init_colors = torch.from_numpy(color).unsqueeze(0).cuda()

        file_name = "record_" + scene_name + "_stage1_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        curves_ctl_pts = json_data['curves_ctl_pts']
        print("Number of curves:", len(curves_ctl_pts))

        start = time.perf_counter()
        chamLoss = dist_chamfer_3D.chamfer_3DDist()

        if curve_type != "cubic":
            curve_type = "cubic"
            curves_ctl_pts_new = Line2Cubic(curves_ctl_pts)
        else:
            curves_ctl_pts_new = curves_ctl_pts
        curve_model = Curves_Model(n_curves=len(curves_ctl_pts), initial_params=torch.tensor(curves_ctl_pts_new),
                                   curve_type=curve_type)

        epoch_stage2 = 2000  # 3000
        lr2 = 0.005  # 0.005
        dis_threshold_2 = 0.02  # 0.02
        alpha = 2  # 2
        loss_end_pts_weight = 0.005  # 0.005
        use_opacity = True
        use_radius = False

        optimizer = torch.optim.Adam(curve_model.parameters(), lr=lr2)
        progress_bar2 = tqdm(range(epoch_stage2), desc="Stage2 progress")

        for iters in range(epoch_stage2):
            pts_curve, pts_curve_m, current_params, params_weight = curve_model(use_weight=use_rational_bezier)        # pts_curve:   [1, N * 100, 3]     pts_curve_m: [1, N * 500, 3]

        # end_pts loss

            end_pts = torch.stack([current_params[:, 0, :], current_params[:, -1, :]], dim=1).reshape(-1, 3)  # torch.Size([n * 2, 3])
            dists = torch.pdist(end_pts, p=2)  # torch.Size([(n * 2) ** 2 / 2 - n])
            mask = torch.ones_like(dists)
            mask[dists > dis_threshold_2] = 0      # 0.01
            masked_dists = dists * mask
            loss_end_pts = loss_end_pts_weight * masked_dists.sum()

        # chamfer loss
            dist1, dist2, idx1, idx2 = chamLoss(pts_curve_m, init_centers_target)       # [1, 500]  [1, N]   [1, 500]   [1, N]
    
            if use_radius:
                dist1[dist1 < radius ** 2] = 1e-12
                dist2[dist2 < radius ** 2] = 1e-12

            if use_opacity:
                chamfer_weight1 = torch.gather(init_opacities, dim=1, index=idx1.long())
                chamfer_weight2 = init_opacities
                chamfer_loss_1 = alpha * torch.sqrt(dist1 * chamfer_weight1).mean()
                chamfer_loss_2 = torch.sqrt(dist2 * chamfer_weight2).mean()
            else:
                chamfer_loss_1 = alpha * torch.sqrt(dist1).mean()
                chamfer_loss_2 = torch.sqrt(dist2).mean()

            loss = chamfer_loss_1 + chamfer_loss_2 + loss_end_pts 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar2.set_postfix({"loss": f"{loss}"}) 
            progress_bar2.update(1)
            if iters == epoch_stage2:
                progress_bar2.close()

        print("Stage 2 time comsumed", time.perf_counter() - start)
        params_weight = params_weight.tolist()
        print("params_weight  max:", np.max(np.array(params_weight)), " min:", np.min(np.array(params_weight)), " mean:", np.mean(np.array(params_weight)))

        json_data = {
            "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "scene_name": scene_name,
            'curves_ctl_pts': current_params.tolist(),
            'curves_ctl_weights': params_weight,
        }

        file_name = "record_" + scene_name + "_stage2_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        print("json file saved in", json_path)

        # save in obj format
        if save_obj_format:
            save_curve_from_json_to_obj(save_curve_dir, file_name, curve_type='cubic')


    if eval_metrics:
        eval_curve_metrics(object_id, save_curve_dir)

