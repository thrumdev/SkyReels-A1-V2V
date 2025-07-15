import json
import pickle
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import os
import torch
import numpy as np
import cv2
from decord import VideoReader
from omegaconf import OmegaConf

def load_manifest(data_dir_path, mode="train"):
    """
    Load the manifest file from the specified directory.
    The manifest is expected to be a JSON file containing a list of items.
    Each item should have a 'filepath' key pointing to the data file.
    """
    manifest_path = f"{data_dir_path}/manifest_{mode}.json"
    with open(manifest_path, "r") as f:
        return json.load(f)

class SkyReelsV2VDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir
        self.manifest = load_manifest(data_dir, mode)

    def __len__(self):
        return len(self.manifest)

    def load_video_tensor(self, path):
        # Load video using decord, expects FFV1 .avi files
        vr = VideoReader(path)
        frames = vr.get_batch(range(len(vr))).asnumpy() # (T, H, W, C)
        frames = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)

        # Crop to 480 height
        if frames.shape[2] > 480:
            height = frames.shape[2]
            crop_height = 480
            crop_t = (height - crop_height) // 2
            frames = frames[:, :, crop_t:crop_t + crop_height, :]

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
        pixel_mask = self.load_video_tensor(mask_path)[0, :, :, :].unsqueeze(0)  # (1, T, H, W)
        flow_mask = self.load_video_tensor(flow_mask_path)[0, :, :, :].unsqueeze(0)  # (1, T, H, W)
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

def list_collate(batch):
    # Returns a dict of lists for each key in the batch
    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    return collated

def get_dataloader(data_dir, config, mode="train"):
    """
    Returns a DataLoader for the given data_dir and config.
    """
    dataset = SkyReelsV2VDataset(data_dir, mode)
    num_workers = config.get("dataloader_workers", 5)
    if mode == "train":
        batch_size = config.batch_size
    else if mode == "val":
        batch_size = config.get("val_batch_size", config.batch_size)    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=1,
        collate_fn=list_collate,
        shuffle=True,
    )
