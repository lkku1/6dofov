import os
import json
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from core.file_client import FileClient
from core.flow_util import resize_flow
from core.img_util import imfrombytes
from RAFT.core.utils.frame_utils import readFlow

from core.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip, GroupRandomHorizontalFlowFlip)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.condition_root = args['condition_root']
        self.num_frames = args['num_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.condition_root)

        self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list

        self.video_names = list(self.video_dict.keys())  # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(div=False),
        ])

        self.file_client = FileClient('disk')
        self.max_pool = torch.nn.MaxPool2d(kernel_size=41, stride=1, padding=20)

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - 1)
        local_idx = complete_idx_set[pivot:pivot + sample_length]

        return local_idx

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name], self.num_frames)

        # read video frames
        frames = []
        depths = []
        con_masks = []
        flows_f, flows_b = [], []
        flows_zeros = np.zeros((self.h, self.w, 2))
        flows_b.append(flows_zeros)
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            dep_path = os.path.join(self.condition_root, video_name + "_depth", frame_list[idx].replace("jpg", "png"))
            img_bytes = self.file_client.get(dep_path, 'img')
            dep = imfrombytes(img_bytes, flag="unchanged", float32=False)
            dep = cv2.resize(dep, self.size, interpolation=cv2.INTER_LINEAR)
            dep = Image.fromarray(dep.astype(np.float64))

            frames.append(img)
            depths.append(dep)
            con_masks.append(all_masks[idx])

            if len(frames) <= self.num_frames - 1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx + 1][:-4]
                flow_f_path = os.path.join(self.condition_root, video_name + "_flow", f'{current_n}to{next_n}.flo')
                flow_b_path = os.path.join(self.condition_root, video_name + "_flow", f'{next_n}to{current_n}.flo')
                flow_f = readFlow(flow_f_path)
                flow_b = readFlow(flow_b_path)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_frames:  # random reverse
                flows_f.append(flows_zeros)
                if random.random() < 0.5:
                    frames.reverse()
                    depths.reverse()
                    con_masks.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_


        if self.load_flow:
            frames, depths, flows_f, flows_b = GroupRandomHorizontalFlowFlip()(frames, depths, flows_f, flows_b)
        else:
            frames = GroupRandomHorizontalFlip()(frames)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames).div(255) * 2 - 1
        depth_tensors = self._to_tensors(depths).div(20 * 300) * 2 - 1
    
        con_mask_tensors = self._to_tensors(con_masks).div(255)
        occ_mask_tensors = self.max_pool(con_mask_tensors) - con_mask_tensors

        # cv2.imwrite("1.png", con_mask_tensors[0,0].numpy()*255)
        # cv2.imwrite("2.png", occ_mask_tensors[0,0].numpy()*255)
        # occ_mask_tensors = self._to_tensors(occ_masks)
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1)  # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # img [-1,1] mask [0,1]
        if self.load_flow:
            return frame_tensors, depth_tensors, con_mask_tensors, occ_mask_tensors, flows_f, flows_b, video_name
        else:
            return frame_tensors, depth_tensors, con_mask_tensors, occ_mask_tensors, 'None', 'None', video_name


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.video_root = args['video_root']
        self.condition_root = args['condition_root']
        self.num_frames = args['num_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.condition_root)

        self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list

        self.video_names = list(self.video_dict.keys())  # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(div=False),
        ])

        self.file_client = FileClient('disk')
        self.max_pool = torch.nn.MaxPool2d(kernel_size=41, stride=1, padding=20)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        depths = []
        masks = []
        flows_zeros = np.zeros((self.h, self.w, 2))
        flows_f, flows_b = [], []
        flows_zeros = np.zeros((self.h, self.w, 2))
        flows_b.append(flows_zeros)
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])

            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)

            dep_path = os.path.join(self.condition_root, video_name + "_depth", frame_list[idx].replace("jpg", "png"))
            img_bytes = self.file_client.get(dep_path, 'img')
            dep = imfrombytes(img_bytes, flag="unchanged", float32=False)
            dep = cv2.resize(dep, self.size, interpolation=cv2.INTER_LINEAR)
            dep = Image.fromarray(dep.astype(np.float64))
            depths.append(dep)

            mask_path = os.path.join("D:/linux/github2", 'mask.png')
            mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')

            masks.append(mask)

            if  idx < len(selected_index)-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx + 1][:-4]
                flow_f_path = os.path.join(self.condition_root, video_name + "_flow", f'{current_n}to{next_n}.flo')
                flow_b_path = os.path.join(self.condition_root, video_name + "_flow", f'{next_n}to{current_n}.flo')
                flow_f = readFlow(flow_f_path)
                flow_b = readFlow(flow_b_path)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if  idx < len(selected_index)-1 and not self.load_flow:
                flows_f.append(flows_zeros)
                flows_b.append(flows_zeros)

            if len(frames) == self.num_frames:  # random reverse
                flows_f.append(flows_zeros)


        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames).div(255) * 2 - 1
        depth_tensors = self._to_tensors(depths).div(20 * 300)
        
        mask_tensors = self._to_tensors(masks)


        flows_f = np.stack(flows_f, axis=-1)  # H W 2 T-1
        flows_b = np.stack(flows_b, axis=-1)
        flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
        flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        return frame_tensors, depth_tensors, mask_tensors, flows_f, flows_b, video_name


if __name__ == '__main__':

    # mask = create_random_shape_with_random_motion(5, 240, 432)[0]
    # img = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    # img = img.view(mask.size[1], mask.size[0], len(mask.mode)).transpose(0, 1).transpose(0, 2).contiguous()[None]
    # max_pool = torch.nn.MaxPool2d(kernel_size=41, stride=1, padding=20)
    # imgg = max_pool(img) - img
    # cv2.imwrite("1.png", img[0, 0].numpy())
    # cv2.imwrite("2.png", imgg[0, 0].numpy())
    # path = "/media/lyb/CE7258D87258C73D/linux/github2/DAVIS-data/DAVIS/JPEGImages/480p_condition/0bebb693b9_depth/00165.png"
    # img_bytes = FileClient('disk').get(path, 'img')
    # deps = []
    # dep = imfrombytes(img_bytes, flag="unchanged", float32=False)
    # dep = cv2.resize(dep, (432, 240), interpolation=cv2.INTER_LINEAR)
    # dep = Image.fromarray(dep.astype(np.float64))
    # deps.append(dep)
    # deps.append(dep)
    # if deps[0].mode == "F":
    #     pic = np.stack([np.expand_dims(x, 2) for x in deps], axis=2)
    # pic = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()

    a = cv2.imread("D:/linux/github2/mask.png")
    b_channel, g_channel, r_channel = cv2.split(a)
    # alpha = np.ones(b_channel.shape, dtype=b_channel.dtype) * 0
    alpha = np.where(b_channel>0, 128, 0).astype(np.uint8)

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    cv2.imwrite("1.png", img_BGRA)
    from FlowFormer.core.utils import flow_viz
    from FlowFormer.core.utils.frame_utils import readFlow
    flow_f = readFlow("D:/linux/github2/move_scene/scene_condition/scene8_flow/234to240.flo")
    flow_img = flow_viz.flow_to_image(flow_f)
    cv2.imwrite("2.png", flow_img)
    print(1)