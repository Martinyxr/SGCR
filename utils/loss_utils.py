#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # print("img1:", img1.shape)
    # print("img2:", img2.shape)
    if len(img1.shape) == 3:
        img1 = img1.view(1, 3, img1.shape[1], img1.shape[2])
        img2 = img2.view(1, 3, img2.shape[1], img2.shape[2])
    # img2 = img2.repeat(3, 1, 1)
    #print("img2:", img2.shape)
    # exit(0)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def w_l1loss(network_output, gt):
    # def rgb2gray(img):
    #     gray = img[0] * 0.299 + img[1] * 0.587 + img[2] * 0.114
    #     return gray.unsqueeze(0)
    # gt = gt.repeat(3, 1, 1)
    # print("network_out:", network_output.shape)
    # print("gt:", gt.shape)
    # exit(0)

    def get_mask(gray):
        thresh = 0.3
        mask = torch.zeros_like(gray).cuda()
        num_positive = (torch.sum(gray > thresh)).float().cuda()
        num_negative = (torch.sum(gray <= thresh)).float().cuda()
        # print("num_positive:", num_positive, "num_negative:", num_negative)
        idx = gray > 0.3
        # try:
        mask[idx] = 1.0 * (num_negative + 1) / (num_positive + num_negative + 1e-5)
        mask[~idx] = 1.0 * (num_positive + 1) / (num_positive + num_negative + 1e-5)
        # except:
            # print("network_output:", network_output, "gt:", gt)
            # print("gray:", gray, "thres:", thresh)
            # print("num_negative:", torch.sum(gray > thresh), "num_positive:", torch.sum(gray <= thresh))
            # print("idx:", idx)
        return mask
    # gray_gt = rgb2gray(gt)
    mask = get_mask(gt)
    
    l1_loss = (network_output - gt)**2
    w_l1loss = (l1_loss * mask).mean()
    
    return w_l1loss

def sparsity_loss_origin(network_output, gt):
    def get_mask(gray):
        thresh = 0.3
        mask = torch.zeros_like(gray)
        mask[gray <= thresh] = 1.0
        
        return mask
    
    mask = get_mask(gt)
    
    cauthy_loss = torch.log(1 + torch.square(network_output) / 0.5)
    
    masked_loss = cauthy_loss * mask
    
    return masked_loss.mean()


def sparsity_loss(opacity, s=0.5):
    loss = torch.mean(torch.log(1 + torch.square(opacity) / s))
    return loss

def sparsity_loss_gray(rgb, s=0.5):
    # gray = torch.amax(rgb, dim=-1).unsqueeze(-1)
    gray = rgb.mean(dim=-1).unsqueeze(-1)
    # print(gray.shape)
    loss = torch.mean(torch.log(1 + torch.square(gray) / s))
    return loss

def scaler_loss_origin(scales):
    scales = torch.abs(scales)
    s, _ = torch.sort(scales, dim=1)
    s0 = s[:, 0]
    s1 = s[:, 1]
    # sorted_s = s[:, :2]
    # scaler_loss = torch.sum(sorted_s, dim=1).mean()
    
    loss0 = torch.log(1 + torch.square(s0) / 0.5)
    loss1 = torch.log(1 + torch.square(s1) / 0.5)
    scaler_loss0 = (loss0 + loss1).mean()
    scaler_loss = s0 + s1
    return scaler_loss


def scaler_loss(scales):
    scales = torch.abs(scales)
    s, _ = torch.sort(scales, dim=1)
    s0 = s[:, 0]
    s1 = s[:, 1]
    s2 = s[:, 2]
    # sorted_s = s[:, :2]
    # scaler_loss = torch.sum(sorted_s, dim=1).mean()

    # ratio = 0.01 * torch.clip(s1 / s2, 0, 1)

    loss0 = torch.log(1 + torch.square(s0) / 0.5)
    # loss1 = torch.log(1 + torch.square(s1) / 0.5)
    # loss2 = torch.log(1 + torch.square(s2) / 0.5)

    # scaler_loss = (loss0 + loss1).mean()
    scaler_loss = loss0.mean()
    # scaler_loss = ratio.mean()
    return scaler_loss


def consisy_loss(precompute_color, opacity):
    # return torch.abs((precompute_color - opacity)).mean()
    return torch.mean((opacity - precompute_color)**2)

