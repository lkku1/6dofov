import numpy as np
import argparse
import glob
import os
from tqdm import tqdm
import yaml
import cv2
import time
# from mesh import write_ply
from util import get_samples_video, get_samples_image, read_depth, depth_resize, run_videotoimage, run_depflow, run_masks
import torch
import PIL.Image as Image

os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='D:/linux/github2/videos')
parser.add_argument('--format', type=str, default='mp4')
parser.add_argument('--width', type=int, default=432)
parser.add_argument('--height', type=int, default=240)
parser.add_argument('--image_path', type=str, default='D:/linux/github2/static_panorama/480p')
parser.add_argument('--save_path', type=str, default='D:/linux/github2/static_panorama/480p_condition')
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

'''
# video name (include name)
video_list = get_samples_video(args.path, args.format)

# convert all videos to images
for idx in tqdm(range(len(video_list))):
    depth = None
    video = video_list[idx]
    print("Start")
    print("video to images")
    print("video folder: " + video['video_file'])
    # video split to images and build corresponding folder(color, depth, flow etc)
    run_videotoimage(args.path, video['video_file'], video['video_format'], args.image_path, args.width, args.height)

'''
# Calculate the depth, optical flow and semantics of images in each folder
args.image_path = "D:/linux/github2/result/scene/Matterport3D2"
image_list_set = sorted([file for file in glob.glob(os.path.join(args.image_path, '*'))])

for idx in range(len(image_list_set)):
    '''
    base_name = os.path.basename(image_list_set[idx])
    names = os.listdir(image_list_set[idx])
    min_name = names[0].split(".")[0]
    for file in names:
        name = file.split(".")[0]
        format = file.split(".")[1]
        if name != "background":
            path = os.path.join(image_list_set[idx], name + "." + format)
            to_path = os.path.join(image_list_set[idx], str(int(name) - int(min_name)).zfill(5) +"."+ format)
            os.rename(path, to_path)
    '''
    image_path = os.path.join(args.image_path, image_list_set[idx])
    image = Image.open(image_path)

    png_name = os.path.join(args.image_path, image_list_set[idx][:-4] + ".png")
    image = image.convert('RGB')
    image =image.resize((432, 240))
    # image.save("1.png")
    image.save(png_name)



    # compute depth and flow in per frame, background RGB and depth,

    # base_name = os.path.basename(image_list_set[idx])
    # run_depflow(base_name, args.image_path, args.save_path, args.width, args.height, 2)
    # if args.start == 0:
    #     if idx < int(len(image_list_set) / 3):
    #         run_depflow(base_name, args.image_path, args.width, args.height, 0)
    #

    # base_name = os.path.basename(image_list_set[6])
    # run_depflow(base_name, args.image_path, args.save_path, args.width, args.height, 1)

    # if args.start == 1:
    #     if idx % 2 == 0:
    #         base_name = os.path.basename(image_list_set[idx])
    #         run_depflow(base_name, args.image_path, args.save_path, args.width, args.height, 1)
    # else:
    #     if idx % 2 != 0:
    #         base_name = os.path.basename(image_list_set[idx])
    #         run_depflow(base_name, args.image_path, args.save_path, args.width, args.height, 2)


