# OUR
from utils import ImageDataset, generatemask, read_image

import os
import torch
import cv2
import numpy as np
import math
import torch.nn.functional as F
import argparse
import warnings
from utils import write_depth
import scipy.signal as ss
from PIL import Image
from PIL import ImageChops

warnings.simplefilter('ignore', np.RankWarning)

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    b, c, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid = torch.unsqueeze(grid, dim=0).repeat(b, 1, 1, 1)
    grid.requires_grad = False

    grid_flowx = (grid[:, :, :, 0] + flow[:, :, :, 0]) % w
    grid_flowy = (grid[:, :, :, 1] + flow[:, :, :, 1])

    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flowx / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flowy / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), 3)
    output = torch.nn.functional.grid_sample(x,
                                             grid_flow,
                                             mode=interpolation,
                                             padding_mode=padding_mode,
                                             align_corners=align_corners)
    return output

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def run(option):

    # Generate mask used to smoothly blend the local patch estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    mask_org = generatemask((2000, 2000))
    mask = mask_org.copy()

    reff_depth = cv2.imread(os.path.join(option.data_dir + '_depth', 'background.png'), -1)
    # reff_depth = cv2.resize(reff_depth, dsize=(option_.width, option_.height), interpolation=cv2.INTER_NEAREST)
    sky_mask = cv2.imread(os.path.join(option.data_dir + '_sem', 'background.png'), 0)
    # sky_mask = cv2.resize(sky_mask, dsize=(option_.width, option_.height))
    sky_mask = np.where(sky_mask == 255, 1.0, 0.0)
    dilate_length = int(option_.height / 12)
    correct_depths = []
    num_frame = 90
    for seq_name in range(0, num_frame, 1):
        reff_depth_copy = reff_depth.copy()
        move_mask = cv2.imread(os.path.join(option.data_dir + '_movemask', str(seq_name) + '.png'), 0)
        move_mask = cv2.resize(move_mask, dsize=(option_.width, option_.height),interpolation=cv2.INTER_NEAREST)
        move_mask = np.where(move_mask > 0, 1.0, 0.0)
        depth = cv2.imread(os.path.join(option.data_dir + '_depth', str(seq_name) + '.png'), -1)
        depth = cv2.resize(depth, dsize=(option_.width, option_.height), interpolation=cv2.INTER_NEAREST)

        ln, li, st, _ = cv2.connectedComponentsWithStats(move_mask.astype(np.uint8), 8, cv2.CV_32S)

        for index in range(ln):
            if index != 0:
                index_mask = np.where(li == index, 1, 0)
                index_dilate = cv2.dilate(index_mask.astype(np.uint8),
                                          np.ones((dilate_length, dilate_length), np.uint8),
                                          iterations=1)
                dilate_region = ((index_dilate - index_mask) * (1 - sky_mask)).astype(bool)
                if dilate_region.any() == False:
                    dilate_region = ((index_dilate - index_mask)).astype(bool)
                p_coef = np.polyfit(depth[dilate_region], reff_depth[dilate_region], deg=1)
                depth_correct = np.polyval(p_coef, depth.reshape(-1)).reshape(depth.shape)

                depth_correct = np.clip(depth_correct, 0, 2 ** 16 - 1)

                box_min_w = st[index][0]
                box_max_w = st[index][0] + st[index][2]
                box_min_h = st[index][1]
                box_max_h = st[index][1] + st[index][3]
                gauss_mask = cv2.resize(mask_org, dsize=(st[index][2], st[index][3]))
                box_index_mask = index_mask[box_min_h:box_max_h, box_min_w:box_max_w]
                gauss_mask = np.where(box_index_mask == 1, 1, gauss_mask)
                fusion_result = reff_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w] * (
                            1 - gauss_mask) + depth_correct[box_min_h:box_max_h, box_min_w:box_max_w] * gauss_mask
                reff_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w] = np.where(
                    fusion_result < reff_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w],
                    reff_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w], fusion_result)

        correct_depths.append(np.float32(reff_depth_copy))

    for seq_name in range(0, num_frame, 1):
        flow_1_path = os.path.join(option.data_dir + '_flow', str(seq_name) + 'to' + str(max(seq_name - 12, 0)) + '.flo')
        if os.path.exists(flow_1_path):
            flow_1 = readFlow(flow_1_path)
        else:
            flow_1 = np.zeros((option_.height, option_.width, 2))
        flow_2_path = os.path.join(option.data_dir + '_flow', str(seq_name) + 'to' + str(max(seq_name - 6, 0)) + '.flo')
        if os.path.exists(flow_2_path):
            flow_2 = readFlow(flow_2_path)
        else:
            flow_2 = np.zeros((option_.height, option_.width, 2))
        flow_3_path = os.path.join(option.data_dir + '_flow', str(seq_name) + 'to' + str(max(seq_name + 6, num_frame-1)) + '.flo')
        if os.path.exists(flow_3_path):
            flow_3 = readFlow(flow_3_path)
        else:
            flow_3 = np.zeros((option_.height, option_.width, 2))
        flow_4_path = os.path.join(option.data_dir + '_flow', str(seq_name) + 'to' + str(min(seq_name + 12, num_frame-1)) + '.flo')
        if os.path.exists(flow_4_path):
            flow_4 = readFlow(flow_4_path)
        else:
            flow_4 = np.zeros((option_.height, option_.width, 2))

        flow1 = torch.from_numpy(np.float16(flow_1)).view(1, option_.height, option_.width, 2)
        flow2 = torch.from_numpy(np.float16(flow_2)).view(1, option_.height, option_.width, 2)
        flow3 = torch.from_numpy(np.float16(flow_3)).view(1, option_.height, option_.width, 2)
        flow4 = torch.from_numpy(np.float16(flow_4)).view(1, option_.height, option_.width, 2)

        flow = torch.cat([flow1, flow2, flow3, flow4], dim=0)
        img1 = torch.from_numpy(correct_depths[max(seq_name - 12, 0)]).view(1, option_.height, option_.width, 1).permute(0, 3, 1,2)
        img2 = torch.from_numpy(correct_depths[max(seq_name - 6, 0)]).view(1, option_.height, option_.width, 1).permute(0, 3, 1,2)
        img3 = torch.from_numpy(correct_depths[min(seq_name + 6, num_frame-1)]).view(1, option_.height, option_.width, 1).permute(0, 3, 1,2)
        img4 = torch.from_numpy(correct_depths[min(seq_name + 12, num_frame-1)]).view(1, option_.height, option_.width, 1).permute(0, 3, 1,2)
        img = torch.cat([img1, img2,img3, img4], dim=0)

        img = flow_warp(img, flow).numpy()

        move_mask = cv2.imread(os.path.join(option.data_dir + '_movemask', str(seq_name) + '.png'), 0)
        move_mask = cv2.resize(move_mask, dsize=(option_.width, option_.height))
        move_mask = np.where(move_mask > 0, 1.0, 0.0)
        img = img.sum(0)[0] / 4
        ln, li, st, _ = cv2.connectedComponentsWithStats(move_mask.astype(np.uint8), 8, cv2.CV_32S)
        index_depth_copy = correct_depths[seq_name].copy()
        for index in range(ln):
            if index != 0:
                index_mask = np.where(li == index, 1, 0)

                dilate_region = index_mask.astype(bool)

                p_coef = np.polyfit(index_depth_copy[dilate_region], img[dilate_region], deg=1)
                depth_correct = np.polyval(p_coef, index_depth_copy.reshape(-1)).reshape(depth.shape)

                depth_correct = np.clip(depth_correct, 0, 2 ** 16 - 1)

                box_min_w = st[index][0]
                box_max_w = st[index][0] + st[index][2]
                box_min_h = st[index][1]
                box_max_h = st[index][1] + st[index][3]
                gauss_mask = cv2.resize(mask_org, dsize=(st[index][2], st[index][3]))
                box_index_mask = index_mask[box_min_h:box_max_h, box_min_w:box_max_w]
                gauss_mask = np.where(box_index_mask == 1, 1, gauss_mask)
                fusion_result = index_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w] * (
                            1 - gauss_mask) + depth_correct[box_min_h:box_max_h, box_min_w:box_max_w] * gauss_mask
                index_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w] = np.where(
                    fusion_result < index_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w],
                    index_depth_copy[box_min_h:box_max_h, box_min_w:box_max_w], fusion_result)


        # depth_final = np.where(move_mask==1, img, correct_depths[seq_name])
        depth_final = index_depth_copy / (2 ** 16 - 1) * 255

        reff_depth_copy = cv2.resize(depth_final.astype("uint8"), dsize=(432, 240), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(option.output_dir, str(seq_name) + '.png'), reff_depth_copy)




    print("depth finished")



if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="/media/lyb/CE7258D87258C73D/linux/github2/test_image/cafeteria", type=str)
    parser.add_argument('--output_dir', default="/media/lyb/CE7258D87258C73D/linux/github2/test_image/cafeteria_depth1", type=str)
    parser.add_argument('--width', default=432, type=int)
    parser.add_argument('--height', default=240, type=int)
    # Check for required input
    option_, _ = parser.parse_known_args()

    # Run pipeline
    run(option_)

