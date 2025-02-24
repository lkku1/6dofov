import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from E2P import equi2pers
from P2E import pers2equi
from core.utils import flow_viz
from core.utils import frame_utils
from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate
import cv2
from core.datasets import OurDate
from core.utils.frame_utils import readFlow


Fov = (120, 120)
flow_size = 240
Nrows = 2
device = None

def uv_coor(flow, grid, width, height, flow_size, alpha):

    # 12, 2, heigth, widht
    # u is width, v is height
    # 0 is width, 1 is height

    # Pixel displacement coordinates
    yy, xx = torch.meshgrid(torch.linspace(0, flow_size-1, flow_size), torch.linspace(0, flow_size-1, flow_size))
    screen_points = torch.unsqueeze(torch.stack([xx.flatten(), yy.flatten()], -1).reshape(flow_size, flow_size, 2), dim=0).to(flow.device)
    screen_points = torch.repeat_interleave(screen_points.permute(0, 3, 1, 2), 12, dim=0)
    screen_points = torch.clamp((screen_points + flow) / (flow_size - 1) * 2 - 1, min=-1, max=1)
    # Pixel spherical coordinates

    target_uv = F.grid_sample(grid, screen_points.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
    target_uv = torch.unsqueeze(target_uv, dim=0).permute(0, 2, 3, 4, 1)
    grid = torch.unsqueeze(grid, dim=0).permute(0, 2, 3, 4, 1)
    # Pixel angular distance
    flow_uv = target_uv - grid
    target_uv_equi = torch.cat([flow_uv, alpha], dim=1)

    flow = pers2equi(target_uv_equi, Fov, Nrows, (flow_size, flow_size), (height, width), "Patch_P2E" + str(height))
    # flow = flow[0].permute(1, 2, 0).cpu().numpy()
    # flow_img = flow_viz.flow_to_image(flow)
    # cv2.imwrite("1.png", flow_img)
    flow[:, 0] = flow[:, 0] * width / 2
    flow[:, 1] = flow[:, 1] * height / 2
    return flow

@torch.no_grad()
def validate_sintel(model, input_floder, output_floder, width, height, alpha, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()


    test_dataset = OurDate(image_root=input_floder, width=width, height=height)
    for test_id in range(len(test_dataset)):
        image1, image2, name1, name2 = test_dataset[test_id]
        if not os.path.exists(os.path.join(output_floder, str(name1) + 'to' + str(name2) + '.flo')) or \
                not os.path.exists(os.path.join(output_floder, str(name2) + 'to' + str(name1) + '.flo')) or \
            not os.path.exists(os.path.join(output_floder, str(name1) + 'to' + str(name2) + '.png')) or \
            not os.path.exists(os.path.join(output_floder, str(name2) + 'to' + str(name1) + '.png')):
        # if True:

            image1 = torch.unsqueeze(image1, dim=0).to(device)
            image2 = torch.unsqueeze(image2, dim=0).to(device)
            #

            image1, grid = equi2pers(image1, Fov, Nrows, flow_size)
            image2, _ = equi2pers(image2, Fov, Nrows, flow_size)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_up = uv_coor(flow_up, grid, width, height, flow_size, alpha)
            forward_flow = flow_up[0].permute(1, 2, 0).cpu().numpy()

            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            flow_up = uv_coor(flow_up, grid, width, height, flow_size, alpha)
            backward_flow = flow_up[0].permute(1, 2, 0).cpu().numpy()

            output_file = os.path.join(output_floder, name1 + "to" + name2 + ".flo")
            frame_utils.writeFlow(output_file, forward_flow)
            flow_img = flow_viz.flow_to_image(forward_flow)
            cv2.imwrite(os.path.join(output_floder, name1 + "to" + name2 + ".png"), flow_img)

            output_file = os.path.join(output_floder, name2 + "to" + name1 + ".flo")
            frame_utils.writeFlow(output_file, backward_flow)
            flow_img = flow_viz.flow_to_image(backward_flow)
            cv2.imwrite(os.path.join(output_floder, name2 + "to" + name1 + ".png"), flow_img)

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.05*size[0]):size[0] - int(0.05*size[0]), int(0.05*size[1]): size[1] - int(0.05*size[1])] = 1

    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="D:/linux/github2/3dp/RAFT/chickpoint/raft-things.pth")
    parser.add_argument('--input_floder', default="D:/linux/github2/Omni/Omni/002")
    parser.add_argument('--output_floder',default="D:/linux/github2/Omni/Omni_condition/002_flow")
    parser.add_argument('--width', default=432, type=int)
    parser.add_argument('--height', default=240, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    input_ = os.listdir(args.input_floder)
    if "background.jpg" in input_:
        input_.remove("background.jpg")

    if (len(input_) - 1) * 4 != len(os.listdir(args.output_floder)):
    # if True:

        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        device = torch.device("cuda:" + str(args.gpu))
        model.to(device)
        model.eval()

        # create_sintel_submission(model.module, warm_start=True)
        # create_kitti_submission(model.module)
        alpha = torch.tensor(generatemask((240, 240)).reshape(1, 1, 240, 240, 1))
        alpha = torch.repeat_interleave(alpha, 12, dim=-1).to(device)
        with torch.no_grad():
            validate_sintel(model.module, args.input_floder, args.output_floder, args.width, args.height, alpha)

    print("flow finish")

