import json
import pickle
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import os
import torch
import numpy as np
import cv2
from decord import VideoReader

def load_manifest(data_dir_path):
    """
    Load the manifest file from the specified directory.
    The manifest is expected to be a JSON file containing a list of items.
    Each item should have a 'filepath' key pointing to the data file.
    """
    manifest_path = f"{data_dir_path}/manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)

class BucketBatchSampler(Sampler):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.get("batch_size", 1)
        self.manifest = load_manifest(config.data_dir)

        self.resolution_buckets = {}
        for idx, item in enumerate(self.manifest):
            resolution = item.get("resolution", "unknown")
            if resolution not in self.resolution_buckets:
                self.resolution_buckets[resolution] = []
            self.resolution_buckets[resolution].append(idx)

    def __iter__(self):
        batches = []
        for resolution, indices in self.resolution_buckets.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue
                batches.append(batch_indices)

        random.shuffle(batches)
        return iter(batches)


class SkyReelsV2VDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.manifest = load_manifest(config.data_dir)
        self.data_dir = config.data_dir

    def __len__(self):
        return len(self.manifest)

    def load_video_tensor(self, path):
        # Load video using decord, expects FFV1 .avi files
        vr = VideoReader(path)
        frames = vr.get_batch(range(len(vr))).asnumpy() # (T, H, W, C)
        frames = torch.tensor(frames.permute(3, 0, 1, 2)).float() / 255.0 # (C, T, H, W)
        return frames

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        An item, for this dataset, consists of 5 components:
            - ref_video: The reference video tensor loaded from "{video_id}_original.avi" (C, T, H, W)
            - driving_video: The driving video tensor loaded from "{video_id}_driving.avi" (C, T, H, W)
            - mask: The mask video tensor loaded from "{video_id}_mask.avi" (1, T, H, W)
            - optical_flow_mask: The optical flow mask tensor loaded from "{video_id}_flow_mask.avi" (1, T-2, H, W)
            - cropped_aligned_identity: The identity image loaded from "{video_id}_identity.png" (H, W, C) with values in [0, 255]

        All components are loaded from separate files using decord (for videos) and cv2 (for the identity image).
        """
        item_info = self.manifest[idx]
        video_id = item_info["video_id"]
        driving_path = os.path.join(self.data_dir, f"{video_id}_driving.avi")
        original_path = os.path.join(self.data_dir, f"{video_id}_original.avi")
        mask_path = os.path.join(self.data_dir, f"{video_id}_mask.avi")
        flow_mask_path = os.path.join(self.data_dir, f"{video_id}_flow_mask.avi")
        identity_path = os.path.join(self.data_dir, f"{video_id}_identity.png")
        driving_video = self.load_video_tensor(driving_path)
        original_video = self.load_video_tensor(original_path)
        pixel_mask = self.load_video_tensor(mask_path)
        flow_mask = self.load_video_tensor(flow_mask_path)
        identity_image = cv2.imread(identity_path)
        identity_image = cv2.cvtColor(identity_image, cv2.COLOR_BGR2RGB)
        identity_image = torch.tensor(identity_image).permute(2, 0, 1)  # (C, H, W)
        return {
            "ref_video": original_video,
            "driving_video": driving_video,
            "mask": pixel_mask,
            "optical_flow_mask": flow_mask,
            "cropped_aligned_identity": identity_image
        }


def get_dataloader(config):
    dataset = SkyReelsV2VDataset(config)
    return DataLoader(dataset, batch_sampler=BucketBatchSampler(config))
