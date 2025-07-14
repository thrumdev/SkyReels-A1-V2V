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

class BucketBatchSampler(Sampler):
    def __init__(self, config, mode="train", world_size=1, rank=0, seed=42):
        self.config = config
        self.batch_size = config.get("batch_size", 1)
        self.manifest = load_manifest(config.data_dir)
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0  # For per-epoch shuffling

        self.resolution_buckets = {}
        for idx, item in enumerate(self.manifest):
            resolution = item.get("resolution", "unknown")
            if resolution not in self.resolution_buckets:
                self.resolution_buckets[resolution] = []
            self.resolution_buckets[resolution].append(idx)

    def __iter__(self):
        # Bump epoch for each call to __iter__
        self.epoch += 1
        rng = random.Random(self.seed + self.epoch)
        batches = []

        # Deterministic loop.
        for resolution in sorted(self.resolution_buckets.keys()):
            indices = self.resolution_buckets[resolution].copy()
            rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue
                batches.append(batch_indices)

        rng.shuffle(batches)
        # Shard batches for distributed training
        total_batches = len(batches)
        # Ensure all replicas yield the same number of batches
        # Note this discards the remainder if total_batches is not divisible by world_size
        # This should balance out across many epochs.
        num_batches_per_replica = total_batches // self.world_size
        start = self.rank * num_batches_per_replica
        end = start + num_batches_per_replica
        batches = batches[start:end]
        return iter(batches)


class SkyReelsV2VDataset(Dataset):
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.manifest = load_manifest(data_dir)
        self.device = device

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

        return frames.to(self.device)

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

def get_dataloader(data_dir, config, device, mode="train", world_size=1, rank=0, seed=42):
    """
    Returns a DataLoader for the given data_dir and config.
    """
    dataset = SkyReelsV2VDataset(data_dir, device)
    return DataLoader(
        dataset,
        batch_sampler=BucketBatchSampler(config, mode="mode", world_size=world_size, rank=rank, seed=seed),
        collate_fn=list_collate
    )
