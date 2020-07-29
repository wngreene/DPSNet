# Copyright 2020 Massachusetts Institute of Technology
#
# @file stereo_sequence_folder.py
# @author W. Nicholas Greene
# @date 2020-07-29 13:55:45 (Wed)

import numpy as np

import torch.utils.data as tud

from stereo_dataset.gta_sfm_dataset import GTASfMStereoDataset

class StereoSequenceFolder(tud.Dataset):
    def __init__(self, stereo_dataset, transform=None):
        self.stereo_dataset = stereo_dataset
        self.transform = transform
        return

    def __len__(self):
        return len(self.stereo_dataset)

    def __getitem(self, idx):
        sample = self.stereo_dataset[idx]

        tgt_img = np.array(sample["left_image"]).astype(np.float32)
        ref_imgs = [np.array(sample["right_image"]).astype(np.float32)]
        T_left_in_right = np.linalg.inverse(sample["T_right_in_left"])
        ref_poses = [T_left_in_right[:3, :].reshape((1, 3, 4))]
        intrinsics = sample["K"]
        tgt_depth = sample["left_depthmap_true"]

        if self.transform is not None:
            imgs, tgt_depth, intrinsics = self.transform([tgt_img] + ref_imgs, tgt_depth, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        return tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth
