import json
import pickle
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import os
import torch

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

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        An item, for this dataset, consists of 4 components:
            - ref_video: The reference video tensor. (C, T, H, W)
            - driving_video: The driving video tensor (landmarks only). (C, T, H, W)
            - mask: The mask video tensor. (1, T, H, W)
            - optical_flow_mask: The optical flow mask tensor for the reference video (1, T-1, H, W)
              This is precomputed according to section 4.2 of the SkyReels A1 paper.
            - cropped_aligned_identity: (H, W). The cropped and aligned facial identity image tensor.

        All 4 of these components are tensors, pickle encoded together in a single file.
        The manifest contains the paths to these files as a `key->filepath pair`.
        """
        item_info = self.manifest[idx]
        file_path = os.path.join(self.config.data_dir, item_info["filepath"])
        with open(file_path, "rb") as f:
            data = torch.load(f)
        return {
            "ref_video": data["ref_video"],
            "driving_video": data["driving_video"],
            "mask": data["mask"],
            "optical_flow_mask": data["optical_flow_mask"],
            "cropped_aligned_identity": data["cropped_aligned_identity"]
        }


def get_dataloader(config):
    dataset = SkyReelsV2VDataset(config)
    return DataLoader(dataset, batch_sampler=BucketBatchSampler(config))
