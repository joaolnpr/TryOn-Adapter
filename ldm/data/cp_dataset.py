# coding=utf-8
import os

import PIL
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json

import random
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
    
def mask2bbox(mask):
    # Handle empty mask
    if np.sum(mask) == 0:
        return (0, mask.shape[0], 0, mask.shape[1])
    
    # Find non-zero coordinates
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return (0, mask.shape[0], 0, mask.shape[1])
    
    up = np.max(coords[0])
    down = np.min(coords[0])
    left = np.min(coords[1])
    right = np.max(coords[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    
    # Ensure valid bounds
    up = min(up, mask.shape[0])
    down = max(0, min(down, up))
    right = min(right, mask.shape[1])
    left = max(0, min(left, right))
    
    return (down, up, left, right)

class CPDataset(data.Dataset):
    """
        Dataset for CP-VTON.
    """

    def __init__(self, dataroot, image_size=512, mode='train', semantic_nc=13, unpaired=False):
        super(CPDataset, self).__init__()
        # base setting
        self.root = dataroot
        self.unpaired = unpaired
        self.datamode = mode  # train or test or self-defined
        self.data_list = mode + '_pairs.txt'
        self.fine_height = image_size
        self.fine_width = 384 #int(image_size / 256 * 256)
        self.semantic_nc = semantic_nc
        self.data_path = osp.join(dataroot, mode)
        self.crop_size = (self.fine_height, self.fine_width)
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))
        self.normalize = transforms.Normalize((0.5), (0.5))

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(dataroot, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names

    def name(self):
        return "CPDataset"

    def get_agnostic(self, im, im_parse, pose_data):
        # Ensure all images are the same size
        target_size = (self.fine_width, self.fine_height)
        im = im.resize(target_size, Image.BILINEAR)
        im_parse = im_parse.resize(target_size, Image.NEAREST)
        
        parse_array = np.array(im_parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        # Ensure pose_data is within bounds
        pose_data = np.clip(pose_data, 0, max(self.fine_width, self.fine_height))
        
        # Calculate length safely
        if np.linalg.norm(pose_data[5] - pose_data[2]) > 0 and np.linalg.norm(pose_data[12] - pose_data[9]) > 0:
            length_a = np.linalg.norm(pose_data[5] - pose_data[2])
            length_b = np.linalg.norm(pose_data[12] - pose_data[9])
            point = (pose_data[9] + pose_data[12]) / 2
            pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
            pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
            r = int(length_a / 16) + 1
        else:
            # Default values if pose data is invalid
            r = 5

        # mask torso
        for i in [9, 12]:
            if i < len(pose_data):
                pointx, pointy = pose_data[i]
                pointx, pointy = int(pointx), int(pointy)
                if 0 <= pointx < self.fine_width and 0 <= pointy < self.fine_height:
                    agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        
        # Draw lines safely
        for line_points in [[2, 9], [5, 12], [9, 12]]:
            if all(i < len(pose_data) for i in line_points):
                points = [(int(pose_data[i][0]), int(pose_data[i][1])) for i in line_points]
                if all(0 <= p[0] < self.fine_width and 0 <= p[1] < self.fine_height for p in points):
                    agnostic_draw.line(points, 'gray', width=r * 6)
        
        # Draw polygon safely
        polygon_points = [2, 5, 12, 9]
        if all(i < len(pose_data) for i in polygon_points):
            points = [(int(pose_data[i][0]), int(pose_data[i][1])) for i in polygon_points]
            if all(0 <= p[0] < self.fine_width and 0 <= p[1] < self.fine_height for p in points):
                agnostic_draw.polygon(points, 'gray', 'gray')

        # mask neck
        if 1 < len(pose_data):
            pointx, pointy = pose_data[1]
            pointx, pointy = int(pointx), int(pointy)
            if 0 <= pointx < self.fine_width and 0 <= pointy < self.fine_height:
                agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

        # mask arms
        if all(i < len(pose_data) for i in [2, 5]):
            points = [(int(pose_data[i][0]), int(pose_data[i][1])) for i in [2, 5]]
            if all(0 <= p[0] < self.fine_width and 0 <= p[1] < self.fine_height for p in points):
                agnostic_draw.line(points, 'gray', width=r * 12)
        
        for i in [2, 5]:
            if i < len(pose_data):
                pointx, pointy = pose_data[i]
                pointx, pointy = int(pointx), int(pointy)
                if 0 <= pointx < self.fine_width and 0 <= pointy < self.fine_height:
                    agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')
        
        for i in [3, 4, 6, 7]:
            if i < len(pose_data) and i-1 < len(pose_data):
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                points = [(int(pose_data[j][0]), int(pose_data[j][1])) for j in [i - 1, i]]
                if all(0 <= p[0] < self.fine_width and 0 <= p[1] < self.fine_height for p in points):
                    agnostic_draw.line(points, 'gray', width=r * 10)
                    pointx, pointy = pose_data[i]
                    pointx, pointy = int(pointx), int(pointy)
                    if 0 <= pointx < self.fine_width and 0 <= pointy < self.fine_height:
                        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # Create mask_arm with the same size as the target
            mask_arm = Image.new('L', target_size, 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            
            if pose_ids[0] < len(pose_data):
                pointx, pointy = pose_data[pose_ids[0]]
                pointx, pointy = int(pointx), int(pointy)
                if 0 <= pointx < self.fine_width and 0 <= pointy < self.fine_height:
                    mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'black', 'black')
            
            for i in pose_ids[1:]:
                if i < len(pose_data) and i-1 < len(pose_data):
                    if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                            pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                        continue
                    points = [(int(pose_data[j][0]), int(pose_data[j][1])) for j in [i - 1, i]]
                    if all(0 <= p[0] < self.fine_width and 0 <= p[1] < self.fine_height for p in points):
                        mask_arm_draw.line(points, 'black', width=r * 10)
                        pointx, pointy = pose_data[i]
                        pointx, pointy = int(pointx), int(pointy)
                        if 0 <= pointx < self.fine_width and 0 <= pointy < self.fine_height:
                            if i != pose_ids[-1]:
                                mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'black', 'black')
                            else:
                                mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black', 'black')

            # Use the already resized parse_array
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            parse_arm_img = Image.fromarray(np.uint8(parse_arm * 255), 'L')
            agnostic.paste(im, None, parse_arm_img)

        # Paste head and lower body
        head_mask = Image.fromarray(np.uint8(parse_head * 255), 'L')
        lower_mask = Image.fromarray(np.uint8(parse_lower * 255), 'L')
        agnostic.paste(im, None, head_mask)
        agnostic.paste(im, None, lower_mask)
        
        return agnostic

    

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = 'image/' + im_name
        c_name = {}
        c = {}
        cm = {}
        if self.unpaired:
            key = 'unpaired'
        else:
            key = 'paired'

        c_name[key] = self.c_names[key][index]
        c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
        c[key] = transforms.Resize(self.crop_size, interpolation=2)(c[key])
        c_img = c[key]
        cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
        cm[key] = transforms.Resize(self.crop_size, interpolation=0)(cm[key])
        cm_img = cm[key]

        c[key] = self.transform(c[key])  # [-1,1]
        cm_array = np.array(cm[key])
        cm_array = (cm_array >= 128).astype(np.float32)
        cm[key] = torch.from_numpy(cm_array)  # [0,1]
        cm[key].unsqueeze_(0)
        # c[key] = c[key] * cm[key]

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name)).convert('RGB')
        im_pil = transforms.Resize(self.crop_size, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # load parsing image
        parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize(self.crop_size, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

        # seg_mask = self.transform(im_parse_pil.convert('RGB'))
        # seg_mask = im_parse_pil.convert('RGB')
        mask = cv2.imread(osp.join(self.data_path, parse_name))  # [:,:,0]
        mask = cv2.resize(mask, (384,512))
        seg_mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        # parse map
        labels = {
            0: ['background', [0]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11, 10]]
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]
                                       
        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        mask_id = torch.Tensor([3, 5, 6, 12])
        mask = torch.isin(parse_onehot[0], mask_id).numpy()

        kernel_size = int(5 * (self.fine_width / 256))
        mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=3)
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=1)
        mask = mask.astype(np.float32)
        inpaint_mask = 1 - self.toTensor(mask)
        if self.datamode == 'test':
            warped_cloth_name = im_name.replace('image', 'test_paired/warped' if not self.unpaired else 'test_unpaired/warped').replace('jpg','png')
        else:
            warped_cloth_name = im_name.replace('image', 'train_paired/warped').replace('jpg','png')

        # Handle missing warped cloth file for single inference
        warped_cloth_path = osp.join(self.data_path, warped_cloth_name)
        if os.path.exists(warped_cloth_path):
            warped_cloth = Image.open(warped_cloth_path)
            warped_cloth = transforms.Resize(self.crop_size, interpolation=2)(warped_cloth)
            warped_cloth = self.transform(warped_cloth)
        else:
            # Create placeholder warped cloth from original cloth
            warped_cloth = c[key].clone()
        
        if self.datamode == 'test':
            warped_cloth_mask_name = im_name.replace('image','test_paired/mask' if not self.unpaired else 'test_unpaired/mask').replace('jpg','png')
        else:
            warped_cloth_mask_name = im_name.replace('image','train_paired/mask').replace('jpg','png')
            
        # Handle missing warped cloth mask file for single inference
        warped_cloth_mask_path = osp.join(self.data_path, warped_cloth_mask_name)
        if os.path.exists(warped_cloth_mask_path):
            warped_cloth_mask = Image.open(warped_cloth_mask_path)
            warped_cloth_mask = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.NEAREST) \
                (warped_cloth_mask)
            warped_cloth_mask = self.toTensor(warped_cloth_mask)
        else:
            # Create placeholder warped cloth mask
            warped_cloth_mask = cm[key].clone()
        
        warped_cloth = warped_cloth * warped_cloth_mask

        feat = warped_cloth * (1 - inpaint_mask) + im * inpaint_mask
        
        #load seg_predicts
        if self.datamode == 'test':
            seg_predicts_name = im_name.replace('image', 'test_paired/seg_preds' if not self.unpaired else 'test_unpaired/seg_preds').replace('jpg','png')
        else:
            seg_predicts_name = im_name.replace('image', 'train_paired/seg_preds').replace('jpg','png')
        
        # Handle missing seg_predicts file for single inference
        seg_predicts_path = osp.join(self.data_path, seg_predicts_name)
        if os.path.exists(seg_predicts_path):
            seg_predicts = Image.open(seg_predicts_path)
            seg_predicts = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.NEAREST) \
                (seg_predicts)
            seg_predicts = self.toTensor(seg_predicts)
        else:
            # Create placeholder seg_predicts from parse image
            seg_predicts = parse_onehot
        
        sobel_img_or =  cv2.imread(osp.join(self.data_path, warped_cloth_name), cv2.IMREAD_GRAYSCALE)
        if sobel_img_or is None:
            # Create placeholder sobel image if warped cloth doesn't exist
            sobel_img_or = np.zeros((512, 384), dtype=np.uint8)
        sobel_img_or = cv2.resize(sobel_img_or,(384,512))
        sobelx = cv2.Sobel(sobel_img_or, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sobel_img_or, cv2.CV_64F, 0, 1, ksize=3)
        kernel = np.ones((3, 3), np.uint8)
        
        youhua_mask = cv2.imread(osp.join(self.data_path, warped_cloth_mask_name), cv2.IMREAD_GRAYSCALE)
        if youhua_mask is None:
            # Create placeholder mask if warped cloth mask doesn't exist
            youhua_mask = np.zeros((512, 384), dtype=np.uint8)
        eroded_mask = cv2.erode(youhua_mask, kernel, iterations=5)
        gradient = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
        gradient[eroded_mask == 0] = 0
        sobel_combined_image = img2tensor(np.expand_dims(gradient, axis=2), bgr2rgb=True, float32=True) / 255.
        # sobel_combined_image = self.normalize(sobel_combined_image)

        # Handle potential empty mask in mask2bbox
        cm_numpy = cm[key][0].numpy()
        if np.sum(cm_numpy) > 0:
            down, up, left, right = mask2bbox(cm_numpy)
            ref_image = c[key][:, down:up, left:right]
        else:
            # If mask is empty, use the whole image
            ref_image = c[key]
        ref_image = (ref_image + 1.0) / 2.0
        ref_image = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)(ref_image)
        ref_image = self.clip_normalize(ref_image)
        
        # load pose image
        pose_name = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')
        pose_path = osp.join(self.data_path, pose_name)
        if os.path.exists(pose_path):
            pose_rgb = Image.open(pose_path)
            pose_rgb = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.BILINEAR)(pose_rgb)
            pose_rgb = self.transform(pose_rgb)  # [-1,1]
        else:
            # Create placeholder pose image (black image)
            pose_rgb = torch.zeros_like(im)

        # load pose points
        pose_name = im_name.replace('image', 'openpose_json').replace('.jpg', '_keypoints.json')
        pose_json_path = osp.join(self.data_path, pose_name)
        if os.path.exists(pose_json_path):
            try:
                with open(pose_json_path, 'r') as f:
                    pose_label = json.load(f)
                    if pose_label['people'] and len(pose_label['people']) > 0:
                        pose_data = pose_label['people'][0]['pose_keypoints_2d']
                        pose_data = np.array(pose_data)
                        pose_data = pose_data.reshape((-1, 3))[:, :2]
                    else:
                        # No people detected, use placeholder
                        pose_data = np.zeros((18, 2))
                        pose_data[:, 0] = self.fine_width // 2
                        pose_data[:, 1] = self.fine_height // 2
            except (json.JSONDecodeError, KeyError, IndexError):
                # JSON parsing failed, use placeholder
                pose_data = np.zeros((18, 2))
                pose_data[:, 0] = self.fine_width // 2
                pose_data[:, 1] = self.fine_height // 2
        else:
            # Create placeholder pose data (centered pose)
            pose_data = np.zeros((18, 2))
            pose_data[:, 0] = self.fine_width // 2  # x coordinates
            pose_data[:, 1] = self.fine_height // 2  # y coordinates
        
        # Ensure pose_data has valid coordinates
        pose_data = np.clip(pose_data, 0, max(self.fine_width, self.fine_height))
        
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.BILINEAR)(agnostic)
        agnostic = self.transform(agnostic)

        # load image-parse-agnostic
        parse_name = im_name.replace('image', 'image-parse-agnostic-v3.2').replace('.jpg', '.png')
        parse_agnostic_path = osp.join(self.data_path, parse_name)
        if os.path.exists(parse_agnostic_path):
            image_parse_agnostic = Image.open(parse_agnostic_path)
            image_parse_agnostic = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.NEAREST)(image_parse_agnostic)
            parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        else:
            # Create placeholder parse agnostic from regular parse
            parse_agnostic = parse.clone()
        
        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        hands_mask = torch.sum(new_parse_agnostic_map[5:7], dim=0, keepdim=True)
        hands_mask = torch.clamp(hands_mask, min=0.0, max=1.0)
        inpaint = feat * (1 - hands_mask) + agnostic * hands_mask
        im_mask = im * inpaint_mask

        result = {
            "GT": im,
            "inpaint_image": inpaint,
            "im_mask": im_mask,
            "inpaint_mask": inpaint_mask,
            "ref_imgs": ref_image,
            'warp_feat': feat,
            'warp_mask': warped_cloth_mask,
            "file_name": self.im_names[index],
            "warp_cloth": warped_cloth,
            "seg_mask": seg_mask,
            'sobel_img': sobel_combined_image,
            "parse_agnostic": seg_predicts,
            "pose": pose_rgb,
            "cloth_mask": cm,
            "cloth":c 
        }
        return result

    def __len__(self):
        return len(self.im_names)


def pre_alignment(c, cm, parse_roi):
    align_factor = 1.0
    w, h = c.size

    # flat-cloth forground & bbox
    c_array = np.array(c)
    cm_array = np.array(cm)
    c_fg = np.where(cm_array != 0)
    t_c, b_c = min(c_fg[0]), max(c_fg[0])
    l_c, r_c = min(c_fg[1]), max(c_fg[1])
    c_bbox_center = [(l_c + r_c) / 2, (t_c + b_c) / 2]
    c_bbox_h = b_c - t_c
    c_bbox_w = r_c - l_c

    # parse-cloth forground & bbox
    parse_roi_fg = np.where(parse_roi != 0)
    t_parse_roi, b_parse_roi = min(parse_roi_fg[0]), max(parse_roi_fg[0])
    l_parse_roi, r_parse_roi = min(parse_roi_fg[1]), max(parse_roi_fg[1])
    parse_roi_center = [(l_parse_roi + r_parse_roi) / 2, (t_parse_roi + b_parse_roi) / 2]
    parse_roi_bbox_h = b_parse_roi - t_parse_roi
    parse_roi_bbox_w = r_parse_roi - l_parse_roi

    # scale_factor & paste location
    if c_bbox_w / c_bbox_h > parse_roi_bbox_w / parse_roi_bbox_h:
        ratio = parse_roi_bbox_h / c_bbox_h
        scale_factor = ratio * align_factor
    else:
        ratio = parse_roi_bbox_w / c_bbox_w
        scale_factor = ratio * align_factor
    paste_x = int(parse_roi_center[0] - c_bbox_center[0] * scale_factor)
    paste_y = int(parse_roi_center[1] - c_bbox_center[1] * scale_factor)

    # cloth alignment
    c = c.resize((int(c.size[0] * scale_factor), int(c.size[1] * scale_factor)), Image.BILINEAR)
    blank_c = Image.fromarray(np.ones((h, w, 3), np.uint8) * 255)
    blank_c.paste(c, (paste_x, paste_y))
    c = blank_c  # PIL Image
    # c.save(os.path.join(cloth_align_dst, cname))

    # cloth mask alignment
    cm = cm.resize((int(cm.size[0] * scale_factor), int(cm.size[1] * scale_factor)), Image.NEAREST)
    blank_cm = Image.fromarray(np.zeros((h, w), np.uint8))
    blank_cm.paste(cm, (paste_x, paste_y))
    cm = blank_cm  # PIL Image
    # cm.save(os.path.join(clothmask_align_dst, cmname))
    return c, cm


if __name__ == '__main__':
    dataset = CPDataset('/home/ock/aigc/vition-HD', 512, mode='train', unpaired=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for data in loader:
        pass
