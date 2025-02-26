import os
import sys
import glob
import cv2
from skimage.transform import resize
import numpy as np
from core import flow_viz
from core import frame_utils
import requests
import torch
import torch.nn.functional as F
# import pietorch

DEPTH_BASE = 'VideoDepth'
FLOW_BASE = 'RAFT'
SEGMENT_BASE = 'Mask2Former/demo'

DEPTH_OUTPUTS = 'depth'
FLOW_OUTPUTS = 'flow'
SEGMENT_OUTPUTS = 'segment'

anaconda_path = "E:\\anaconda3\\Scripts\\activate.bat"
conda_name = "3dp"


def get_samples_video(video_folder, format):
    lines = sorted([os.path.splitext(os.path.basename(xx)) for xx in glob.glob(os.path.join(video_folder, '*' + format))])
    samples = []

    # all video names
    for seq_dir in lines:
        samples.append({})
        sdict = samples[-1]
        sdict["video_file"] = seq_dir[0]
        sdict["video_format"] = seq_dir[1]

    return samples

def get_samples_image(image_folder):
    lines = [os.path.splitext(os.path.basename(xx)) for xx in glob.glob(os.path.join(image_folder, '*' + '.png'))]
    samples = []

    # all video names
    for seq_dir in lines:
        samples.append({})
        sdict = samples[-1]
        sdict["image_file"] = seq_dir[0]
        sdict["image_format"] = seq_dir[1]

    return samples

def read_depth(path, img_name, width, height):
    rgb = cv2.imread(os.path.join(path, 'color', img_name + '.jpg'))
    rgb = cv2.resize(rgb, (width, height))

    depth = np.load(os.path.join(path, 'depth', img_name +'.npy'))
    depth = cv2.resize(depth, (width, height))

    flow = cv2.imread(os.path.join(path, 'flow', img_name + 'jpg'))
    flow = cv2.imresize(flow, (width, height))

    flow_mask = cv2.imread(os.path.join(path, 'flow_mask', img_name + '.jpg'))
    flow_mask = cv2.resize(flow_mask, (width, height))

    return rgb, depth, flow, flow_mask

def depth_resize(depth, origin_size, image_size):
    if origin_size[0] != 0:
        max_depth = depth.max()
        depth = depth / max_depth
        depth = resize(depth, origin_size, order=1, mode='edge')
        depth = depth * max_depth
    else:
        max_depth = depth.max()
        depth = depth / max_depth
        depth = resize(depth, image_size, order=1, mode='edge')
        depth = depth * max_depth

    return depth

def build_folder(folder_path, folder_name):
    if os.path.exists(os.path.join(folder_path, folder_name)):
        print("Exist image folder")
        return True
    else:
        os.makedirs(os.path.join(folder_path, folder_name))
        print("Create image folder")


def run_videotoimage(video_path, video_name, video_format, image_path, width, height):

    build_folder(image_path, video_name)

    # capture video
    cap = cv2.VideoCapture(os.path.join(video_path, video_name + video_format))
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if len(os.listdir(os.path.join(image_path, video_name))) != frame_num:
        # video to image
        for index in range(int(frame_num)):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (width, height))
            cv2.imwrite(os.path.join(image_path, video_name, str(index).zfill(5) + ".jpg"), frame)


def run_move_mask(image_dir, flow_dir, move_mask_dir, width, height,  step=1, N=1):

    # depth align
    lines = os.listdir(image_dir)
    if 'background.jpg' in lines:
        lines.remove('background.jpg')

    # background sub
    backSub = cv2.createBackgroundSubtractorMOG2(history=len(lines), varThreshold=15, detectShadows=True)
    background_color = cv2.imread(os.path.join(image_dir, 'background.jpg'))
    _ = backSub.apply(background_color)
    for seq_name in range(0, len(lines), N):
        # if not os.path.exists(os.path.join(move_mask_dir, str(seq_name) + ".png")):
        if True:
            # seq_name = seq_name % len(lines)
            if seq_name < step:
                flow_f = frame_utils.readFlow(os.path.join(flow_dir, str(seq_name).zfill(5) + 'to' + str(seq_name + step).zfill(5) + '.flo'))
                flow_fmask = flow_viz.flow_to_mask(flow_f, 0.1) / 255
                flow_mask = flow_fmask.astype(np.uint8)
            elif seq_name == len(lines) - step:
                flow_b = frame_utils.readFlow(os.path.join(flow_dir, str(seq_name).zfill(5) + 'to' + str(seq_name - step).zfill(5) + '.flo'))
                flow_bmask = flow_viz.flow_to_mask(flow_b, 0.1) / 255
                flow_mask = flow_bmask.astype(np.uint8)
            else:

                flow_f = frame_utils.readFlow(os.path.join(flow_dir, str(seq_name).zfill(5) + 'to' + str(seq_name + step).zfill(5) + '.flo'))
                flow_fmask = flow_viz.flow_to_mask(flow_f, 0.1) / 255

                flow_b = frame_utils.readFlow(os.path.join(flow_dir, str(seq_name).zfill(5) + 'to' + str(seq_name - step).zfill(5) + '.flo'))
                flow_bmask = flow_viz.flow_to_mask(flow_b, 0.1) / 255
                flow_mask = (flow_fmask.astype(np.uint8) | flow_bmask.astype(np.uint8))
           
            image = cv2.imread(os.path.join(image_dir, str(seq_name).zfill(5) + '.jpg'))
            image = cv2.resize(image, dsize=(width, height))

            fgMask = backSub.apply(image)
            fgMask = cv2.morphologyEx(np.where(fgMask > 60, 1, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

            fgMask = np.where(fgMask + flow_mask > 0, 255, 0)
            ln, li, st, _ = cv2.connectedComponentsWithStats(fgMask.astype(np.uint8), 8, cv2.CV_32S)
            record_mask = np.zeros(flow_mask.shape)
            for index in range(ln):
                if st[index][4] > 10 and index != 0:
                    lii = np.where(li == index, index, 0)
                    record_mask = lii + record_mask
            
            final_mask = np.where(record_mask > 0, 255, 0)
            cv2.imwrite(os.path.join(move_mask_dir, str(seq_name).zfill(5) + ".png"), final_mask)

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.15*size[0]):size[0] - int(0.15*size[0]), int(0.15*size[1]): size[1] - int(0.15*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

def run_background(image_dir, flow_dir, width, height, step=1, N=1):

    # depth align
    lines = os.listdir(image_dir)
    if 'background.jpg' in lines:
        lines.remove('background.jpg')

    # if not os.path.exists(os.path.join(image_dir, 'background.jpg')):
    if True:

        # create background_color and background_mask
        background_color = np.zeros((height, width, 3))
        background_mask = np.zeros((height, width, 3))
        
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=len(lines), varThreshold=15, detectShadows=True)

        for seq_name in range(0, len(lines), N):

            image = cv2.imread(os.path.join(image_dir, str(seq_name).zfill(5) + '.jpg'))
            image = cv2.resize(image, dsize=(width, height))

            
            mog_fgMask = bg_subtractor.apply(image)
            mog_fgMask =  cv2.morphologyEx(np.where(mog_fgMask > 60, 1, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
            mog_fgMask = np.expand_dims(mog_fgMask, axis=2).repeat(3, axis=2)
            if seq_name > 0:
                background_color = background_color + image * (1 - mog_fgMask) / 255
                background_mask = background_mask + (1 - mog_fgMask)

        
        background_color = background_color / (background_mask + 1e-3)
        cv2.imwrite(os.path.join(image_dir, 'background.jpg'), background_color * 255)
        

def flow_warp(x, flow):
    h, w = flow.shape[:2]
    flow[:, :, 0] = (flow[:, :, 0] + np.arange(w)) % w
    flow[:, :, 1] += np.arange(h)
    res = cv2.remap(x, flow, None, cv2.INTER_LINEAR)
    return res

def run_correct_depth(correct_dir, depth_dir, image_dir, move_dir, width, height, N=1):

    # depth align   
    line_depth = os.listdir(depth_dir)
    line_correct = os.listdir(correct_dir)
    
    if len(line_depth) != len(line_correct):

        # create background_color and background_mask
        reff_depth = cv2.imread(os.path.join(depth_dir, 'background.png'), -1)
        
       
        blur_length = int(height / 12)
        if blur_length % 2 == 0:
            blur_length = blur_length + 1
        for seq_name in range(0, len(line_depth) - 1, N):
            # if os.path.exists(os.path.join(correct_dir, str(seq_name) + '.png')):
            #     continue

            reff_depth_copy = reff_depth.copy()
            depth = cv2.imread(os.path.join(depth_dir, str(seq_name).zfill(5) + '.png'), -1)
            move_mask = cv2.imread(os.path.join(move_dir, str(seq_name).zfill(5) + '.png'), 0)
            move_mask = cv2.resize(move_mask, dsize=(width, height))
            move_mask = np.where(move_mask > 0, 1.0, 0.0)
            
            move_mask = cv2.dilate(move_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1)
            region = 1 - move_mask

            p_coef = np.polyfit(depth[region.astype(bool)], reff_depth[region.astype(bool)], deg=1)
            depth_pol = np.polyval(p_coef, depth.reshape(-1)).reshape(depth.shape)

            index_dilate = cv2.GaussianBlur(move_mask, (blur_length, blur_length), 0)
            reff_depth_copy = index_dilate * depth_pol + (1 - index_dilate) * reff_depth_copy

            cv2.imwrite(os.path.join(correct_dir, str(seq_name).zfill(5) + '.png'), reff_depth_copy.astype("uint16"))
            # cv2.imwrite(os.path.join(correct_dir, str(seq_name) + '.png'), reff_depth_copy/reff_depth_copy.max()*255)

        cv2.imwrite(os.path.join(correct_dir, 'background.png'), reff_depth)

def neig_25(x, y, H, W):
    maxtrix = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            maxtrix.append((max(min(x + i, H-1), 0), (y + j) % W))
    return maxtrix

def neig_8(x, y, H, W):
    return [(max(x - 1, 0), (y - 1) % W), (x, (y - 1) % W), (min(x + 1, H-1), (y - 1) % W), (min(x + 1, H-1), y), (min(x + 1, H-1), (y + 1) % W),
            ((x), (y + 1) % W), (max(x - 1, 0), (y + 1) % W), (max(x - 1, 0), y)]

def neig_4(x, y, H, W):
    return [(max(x - 1, 0), y), (min(x + 1, H-1), y), (x, (y - 1) % W), (x, (y + 1) % W)]

def neig_4x2(x, y, H, W):
    return [(max(x - 2, 0), y), (min(x + 2, H-1), y), (x, (y - 2) % W), (x, (y + 2) % W)]

def compute_mask(depth, valid_mask):
    max_depth = depth.max()
    depth_ori = (depth / max_depth * 255).astype(np.uint8)

    H, W = depth.shape
    depth_edge = cv2.Canny(depth_ori, 30, 50) / 255 * valid_mask


    ln, li, st, _ = cv2.connectedComponentsWithStats(depth_edge.astype(np.uint8), 8, cv2.CV_32S)
    depth_edge_reduce = np.zeros((H, W))
    edge_index = 1
    for index in range(1, ln, 1):
        if st[index][-1] > 10:
            depth_edge_reduce = np.where(li == index, edge_index, depth_edge_reduce)
            edge_index = edge_index + 1

    if depth_edge_reduce.sum() == 0:
        return [], depth_edge_reduce

    edge_index_t = 1
    edge_indexs = []
    for index in range(1, edge_index, 1):
        category_map = np.zeros((H, W))
        edge_index_map = np.where(depth_edge_reduce == index, 1, 0)
        edge_index_dilate = cv2.dilate(edge_index_map.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
        neighbor_map = edge_index_dilate - edge_index_map
        pos = np.where(neighbor_map==1)
        for pos_index in range(len(pos[0])):
            x = pos[0][pos_index]
            y = pos[1][pos_index]
            neighbor8 = neig_8(x, y, H, W)

            neig_depth = 0
            neig_depth_num = 0
            for neig8_p in neighbor8:
                if edge_index_map[neig8_p] == 1:
                    neig_depth = neig_depth + depth[neig8_p]
                    neig_depth_num = neig_depth_num + 1
            if depth[x, y] <= neig_depth / neig_depth_num:
                category_map[x, y] = 2 * edge_index_t - 1
            if depth[x, y] > neig_depth / neig_depth_num:
                category_map[x, y] = 2 * edge_index_t

        flag = np.where(category_map == 2 * edge_index_t - 1, True, False)
        near_mean = depth[flag].mean()
        if near_mean > 15 * 300:
            continue
        flag = np.where(category_map == 2 * edge_index_t, True, False)
        far_mean = depth[flag].mean()
        dilate_num = np.ceil(np.arctan(1.5 * 300 * (far_mean - near_mean) / (near_mean * far_mean))/(2 * 3.1415926 / W))
        edge_index_t = edge_index_t + 1
        edge_indexs.append([category_map, dilate_num])

    return edge_indexs, depth_edge_reduce


def write_depth(path, depth, bits=2):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    # if not grayscale:
    #     out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def flood_fill(mask):

    nf_index_maps = []
    for index in range(len(mask)):

        nf_map = mask[index][0]
        dilate_num = int(mask[index][1])
        dilate_num = 1
        near = np.where(nf_map==nf_map.max() -1, 1, 0)
        far = np.where(nf_map==nf_map.max(), 1, 0)

        near_copy = near.copy() + 0

        for dilate_idx in range(max(dilate_num*2, 5)):
            near = cv2.dilate(near.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            near = np.where(near - far > 0, 1, 0)

            far = cv2.dilate(far.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            far = np.where(far - near > 0, 1, 0)

        near_range = cv2.dilate(near_copy.astype(np.uint8), np.ones((2 + dilate_num, 2 + dilate_num), np.uint8), iterations=1)
        near = near * near_range

        nf_index_map = near + far * 2

        nf_index_maps.append(nf_index_map)

    group_mask = []
    group_mask.append(nf_index_maps[0])
    for index in range(1, len(nf_index_maps)):
        flag = False
        for group_index in range(len(group_mask)):
            group_mask_mask = np.where(group_mask[group_index] > 0, 1, 0)
            nf_index_mask = np.where(nf_index_maps[index] > 0, 1, 0)
            if (nf_index_mask * group_mask_mask).sum() == 0:
                group_mask[group_index] = group_mask[group_index] + nf_index_maps[index]
                flag = True
                break
        if flag == False:
            group_mask.append(nf_index_maps[index])
    group_mask = np.stack(group_mask, -1)
    return group_mask

def run_masks(image_dir, depth_dir, move_dir, sem_dir, inpaint_mask,opacity_dir, width, height, step, overflow, N=8):

    valid_mask = np.zeros((height, width))
    valid_mask[height//N:-height//N, :] = 1

    depth = cv2.imread(os.path.join(depth_dir, 'background.png'), -1)
    # sky_mask = cv2.imread(os.path.join(sem_dir, 'background.png'), 0) / 255
    # sky_mask = np.where(sky_mask > 0, 1.0, 0.0)
    # valid_mask = valid_mask * (1 - sky_mask)

    depth_edge, edge_map = compute_mask(depth, valid_mask)


    if len(depth_edge) == 0:
        result = np.zeros((height, width, 1))
        np.save(os.path.join(inpaint_mask, 'background.npy'), result)
    else:
        result = flood_fill(depth_edge)
        np.save(os.path.join(inpaint_mask, 'background.npy'), result)
        edge_map = np.where(edge_map > 0, 1, 0)
        opacity_mask = 1 - cv2.dilate(edge_map.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        opacity_mask = cv2.GaussianBlur(opacity_mask * 255, (5, 5), 1.3)
        cv2.imwrite(os.path.join(opacity_dir, 'background.png'), opacity_mask)

    video_length = len(os.listdir(image_dir))
    for index in range(0, video_length, step):
        # if os.path.exists(os.path.join(write_path, str(index) + '.png')):
        #     continue

        move_mask = cv2.imread(os.path.join(move_dir, str(index) + '.png'), 0) / 255
        sky_mask = cv2.imread(os.path.join(sem_dir, str(index) + '.png'), 0) / 255
        sky_mask = 1 - sky_mask

        depth = cv2.imread(os.path.join(depth_dir, str(index) + '.png'), -1)

        depth_edge, edge_map = compute_mask(depth, valid_mask=sky_mask * move_mask)
        if len(depth_edge) == 0:
            result = np.zeros((height, width, 1))
            np.save(os.path.join(inpaint_mask, str(index) + '.npy'), result)
        else:
            result = flood_fill(depth_edge)
            cv2.imwrite(os.path.join(inpaint_mask, str(index) + '.npy'), result)

            edge_map = np.where(edge_map > 0, 1, 0)
            opacity_mask = 1 - cv2.dilate(edge_map.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
            opacity_mask = cv2.GaussianBlur(opacity_mask * 255, (5, 5), 1.3)
            cv2.imwrite(os.path.join(opacity_dir, str(index) + '.npy'), opacity_mask)



def run_depflow(video_name, image_path, save_path, width, height, gpu):

    # video sub_path
    build_folder(save_path, video_name + "_depth")
    build_folder(save_path, video_name + "_flow")
    build_folder(save_path, video_name + "_movemask")
    build_folder(save_path, video_name + "_sem")
    build_folder(save_path, video_name + "_inpaint_mask")
    build_folder(save_path, video_name + "_correct")
    build_folder(save_path, video_name + "_opacity")

    image_dir = os.path.join(image_path, video_name)
    depth_dir = os.path.join(save_path, video_name + '_depth')
    correct_dir = os.path.join(save_path, video_name + '_correct')
    flow_dir = os.path.join(save_path, video_name + '_flow')
    move_dir = os.path.join(save_path, video_name + '_movemask')
    sem_dir = os.path.join(save_path, video_name + '_sem')
    inpaint_mask_dir = os.path.join(save_path, video_name + '_inpaint_mask')
    opacity_dir = os.path.join(save_path, video_name + '_opacity')

    '''
    if sys.platform.startswith('win'):
        # run depth and optical flow estimation
        os.system(f'cd {FLOW_BASE} && call {anaconda_path} {conda_name} && python evaluate.py --input_floder {image_dir}/  --output_floder {flow_dir}/ --width {width} --height {height} --gpu {gpu}')
        # run segment
        os.system(f'cd {SEGMENT_BASE} && call {anaconda_path} {conda_name} && python demo.py --input {image_dir}/  --output {sem_dir} --opts MODEL.WEIGHTS /media/lyb/CE7258D87258C73D/linux/github2/3dp/Mask2Formermain/demo/model_final_94dc52.pkl')
    elif sys.platform.startswith("linux"):
        os.system(f'cd {FLOW_BASE} && python evaluate.py --input_floder {image_dir}/  --output_floder {flow_dir}/ --width {width} --height {height} --gpu {gpu}')
        os.system(f'cd {SEGMENT_BASE} && python demo.py --input {image_dir}/  --output {sem_dir} --opts MODEL.WEIGHTS /media/lyb/CE7258D87258C73D/linux/github2/3dp/Mask2Formermain/demo/model_final_94dc52.pkl')
    '''
    
    # run background
    # run_background(image_dir, flow_dir, width, height)
    # run_move_mask(image_dir, flow_dir, move_dir, width, height)
    # run depth
    # if sys.platform.startswith('win'):
    #     os.system(f'cd {DEPTH_BASE} && call {anaconda_path} {conda_name} &&  python run_depth_anything.py --data_dir {image_dir}  --output_dir {depth_dir} --width {width} --height {height} --gpu {gpu}')
    # elif sys.platform.startswith("linux"):
    #     os.system(f'cd {DEPTH_BASE} &&  python run_depth_anything.py --data_dir {image_dir}  --output_dir {depth_dir} --width {width} --height {height} --gpu {gpu}')
    # run_correct_depth(correct_dir, depth_dir, image_dir, move_dir, width, height)
    run_masks(image_dir, depth_dir, move_dir, sem_dir, inpaint_mask_dir, opacity_dir, width, height, 1, 20)


def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath



def clean_folder(folder, img_exts=['.png', '.jpg', '.npy']):

    for img_ext in img_exts:
        paths_to_check = os.path.join(folder, f'*{img_ext}')
        if len(glob.glob(paths_to_check)) == 0:
            continue
        print(paths_to_check)
        os.system(f'rm {paths_to_check}')

if __name__ == '__main__':



    from numpy import inf
    from PIL import Image
    from PIL import ImageChops
    import math

    # move_mask = cv2.imread("/media/lyb/CE7258D87258C73D/linux/github2/move_scene/scene2_movemask/207.png", -1)
    # depth = cv2.imread("/media/lyb/CE7258D87258C73D/linux/github2/move_scene/scene2_depth/207.png", -1)
    # depth_edge = compute_mask(depth, valid_mask_temp=move_mask)
    # # cv2.imwrite("1.png", depth_edge * 60)
    # for index2 in range(1, int(depth_edge.max() / 2) + 1, 1):
    #     result = flood_fill(depth_edge, index2, 432, 240, 20, 7)
    #     result1 = np.where(result == 1, 128, 0)
    #     result2 = np.where(result == 2, 255, 0)
    #     if result1.reshape(-1).sum() == 0 or result2.reshape(-1).sum() == 0:
    #         continue
    #     cv2.imwrite("1.png", result1 + result2)
    #
    # cv2.imwrite("1.png", a * 124)
    # a[:35, :, :] = 0
    # a[-35:, :, :] = 0
    # cv2.imwrite("1.png", a)
    # color = cv2.imread("/media/lyb/CE7258D87258C73D/linux/github2/move_scene/scene1/background.jpg", 0)
    # /media/lyb/CE7258D87258C73D/linux/github2/move_scene/scene1_depth/background.png
    # /media/lyb/CE7258D87258C73D/linux/github2/test_scene/cafeteria_depth/background.png
    # depth = cv2.imread("/media/lyb/CE7258D87258C73D/linux/github2/test_scene/shore_depth/0.png", -1)
    flow = frame_utils.readFlow("/media/lyb/CE7258D87258C73D/linux/github2/move_scene/scene2_flow2/219to213.flo")
    fa = flow[:, :, 0]
    fb = flow[:, :, 1]
    flow_mask = flow_viz.flow_to_image(flow)
    cv2.imwrite("1.png", flow_mask)
    flow_mask = np.where(flow_mask > 0, 255, 0)
    # depth = depth * flow_mask
    back_color = Image.open("/media/lyb/CE7258D87258C73D/linux/github2/test_scene/cafeteria/background.jpg")
    index_color = Image.open("/media/lyb/CE7258D87258C73D/linux/github2/test_scene/cafeteria/0.jpg")
    index_color = ImageChops.invert(index_color)
    index_color = Image.blend(back_color, index_color, 0.5).convert('L')
    index_color = np.asarray(index_color).astype(np.float16)
    index_color = abs(index_color - 127)
    mask = np.where(np.where(index_color > 10, 1, 0) + flow_mask > 0, 1, 0)
    ln, li, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
    mask2 = np.zeros(depth.shape)
    index_num = 0
    for index in range(ln):
        if st[index][4] > 400 and index != 0:
            index_num = index_num + 1
            # label_index_image
            lii = np.where(li == index, 1, 0)
            mask2 = lii + mask2
    depth = depth * mask2
    a = compute_mask(depth)
    # flow = frame_utils.readFlow("/media/lyb/CE7258D87258C73D/linux/github2/3dp_images/scene1_flow/0to6.flo")
    # l2f_flow_mask = np.where(flow > 0.5, 1, 0).sum(-1)
    # l2f_flow_mask = np.where(l2f_flow_mask > 0, 1, 0)
    # cv2.imwrite("a.png", l2f_flow_mask*255)
    print("a")