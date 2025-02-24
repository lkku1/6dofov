# OUR
from utils import ImageDataset, generatemask, read_image

from depth_anything_v2.dpt import DepthAnythingV2

import os
import torch
import cv2
import numpy as np
import math
import torch.nn.functional as F
import argparse
import warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from E2P import equi2pers
from P2E import pers2equi
from torchvision.transforms import functional as TF
import pietorch

Fov = (120, 120)
depth_size = 518
Nrows = 2
device = None

def write_depth(path, depth):
    # depth = (depth[0, 0] * 300).cpu().numpy()
    # depth = depth / depth.max() * 255
    # cv2.imwrite("8.png", depth)
    depth = (depth[0, 0] * 300).cpu().numpy().astype("uint16")
    cv2.imwrite(path + ".png", depth)

    # np.save(path, depth)
    return

# warnings.simplefilter('ignore', np.RankWarning)
def depth_estimation(omni_rgb, depth_anything, alpha, device):


    # omni depth init
    height, width, channel = omni_rgb.shape
    add_length = int(width * 0.1)
    omni_rgb_add = np.concatenate([omni_rgb, omni_rgb[:, :add_length, :]], axis=1)

    omni_rgb_add, size = depth_anything.image2tensor(omni_rgb_add, device, 518)
    omni_depth_init = depth_anything.infer_image(omni_rgb_add, size, 20)

    _, weight_width = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, add_length), indexing="ij")
    weight_width = weight_width[None,None].to(omni_depth_init.device)
    omni_depth_init[..., :add_length] = omni_depth_init[..., -add_length:] * (1 - weight_width) + omni_depth_init[..., :add_length] * weight_width
    omni_depth = omni_depth_init[..., :-add_length]


    omni_rgb, size = depth_anything.image2tensor(omni_rgb, device, depth_size)
    omni_depth = torch.nn.functional.interpolate(omni_depth, size=(omni_rgb.shape[2], omni_rgb.shape[3]), mode="area")
    omni_info = torch.cat([omni_rgb, omni_depth], dim=1)
    per_info, center = equi2pers(omni_info, Fov, Nrows, depth_size)
    per_rgb = per_info[:, :3]
    per_dep_init = per_info[:, 3:4]
    per_dep_refine = depth_anything.infer_image(per_rgb, (depth_size, depth_size), 20)

    per_dep_init = F.pad(per_dep_init, (1, 1, 1, 1), 'replicate')
    per_dep_init = per_dep_init.cpu()
    per_dep_refine = per_dep_refine.cpu()
    mask = torch.ones((per_dep_refine.shape[2:]))
    corner_coord = torch.tensor([1, 1])
    for index in range(per_dep_init.shape[0]):
        source = per_dep_init[index].clone()
        target = per_dep_refine[index]

        per_dep_init[index] = pietorch.blend(source, target, mask, corner_coord, True, channels_dim=0)

        # cv2.imwrite("1.png", torch.movedim(source, 0, -1).numpy() * 10)
        # cv2.imwrite("2.png", torch.movedim(target, 0, -1).numpy() * 10)
        # cv2.imwrite("3.png", torch.movedim(per_dep_init[index], 0, -1).numpy() * 10 )
        # print("a")

    per_dep_init = per_dep_init[..., 1:-1, 1:-1].reshape(1, 12, 1, depth_size, depth_size).permute(0, 2, 3, 4, 1)
    per_dep_init = per_dep_init.to(alpha.device)
    per_refine = torch.cat([per_dep_init, alpha], dim=1)
    omni_depth = pers2equi(per_refine, Fov, Nrows, (depth_size, depth_size), (omni_rgb.shape[2], omni_rgb.shape[3]), "Patch_P2E" + str(depth_size))
    omni_depth = torch.nn.functional.interpolate(omni_depth, size=(height, width), mode="area")


    # omni_rgb, size = depth_anything.image2tensor(omni_rgb, device, 518)
    # omni_depth = depth_anything.infer_image(omni_rgb, size, 20)

    return omni_depth


def run(dataset, option, device):

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    global depth_anything
    depth_anything = DepthAnythingV2(**{**model_configs[option.encoder], 'max_depth': option.max_depth})
    depth_anything.load_state_dict(torch.load(option.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()

    # Go through all images in input directory
    print("start processing")

    alpha = torch.tensor(generatemask((depth_size, depth_size)).reshape(1, 1, depth_size, depth_size, 1))
    alpha = torch.repeat_interleave(alpha, 12, dim=-1).to(device)

    for image_ind, images in enumerate(dataset):
        # if not os.path.exists(os.path.join(option.output_dir, str(images.name) + ".png")):
        if True:
            omni_dep = depth_estimation(images.rgb_image, depth_anything, alpha, device)
            # cv2.imwrite("1.png", (omni_dep[0, 0] / omni_dep.max() * 255).cpu().numpy())
            # print(1)
            write_depth(os.path.join(option.output_dir, str(images.name)), omni_dep)

    print("depth finished")


if __name__ == "__main__":

    # b = cv2.imread("D:/linux/github2/3dp/VideoDepth/2.png", -1)
    # c = np.abs(a - b) / np.abs(a - b).max() * 255
    # cv2.imwrite("3.png", c)

    # Adding necessary input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="D:\\linux\\github2\\papar_f\\scene5", type=str)
    parser.add_argument('--output_dir', default="D:\\linux\\github2\\papar_f\\scene5_depth", type=str)
    parser.add_argument('--width', default=432, type=int)
    parser.add_argument('--height', default=240, type=int)
    parser.add_argument('--encoder', default="vitl")
    parser.add_argument('--load-from', type=str, default='D:/linux/github2/3dp/VideoDepth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    parser.add_argument('--gpu', default=1, type=int)
    # Check for required input
    option_, _ = parser.parse_known_args()
    print(option_)

    # select device
    device = torch.device("cuda:" + str(option_.gpu))

    # Create dataset from input images
    dataset_ = ImageDataset(option_.data_dir, option_.width, option_.height)

    # Run pipeline
    run(dataset_, option_, device)

