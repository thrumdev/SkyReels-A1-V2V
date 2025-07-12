import argparse
import os
import json
from decord import VideoReader
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import multiprocessing

from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils  import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from tqdm import tqdm

from .memfof.core.memfof import MEMFOF

MEMFOF_MODEL = "MEMFOF-Tartan-T-TSKH"

# copied from tartan-t-tskh.json
class MEMFOFConfig:
    def __init__(
        self,
        name="Tartan-T-TSKH",
        dataset="TSKH-full",
        val_datasets=None,
        monitor=None,
        use_var=True,
        var_min=0,
        var_max=10,
        pretrain="resnet34",
        radius=4,
        dim=512,
        num_blocks=2,
        iters=4,
        image_size=None,
        scale=1,
        effective_batch_size=32,
        num_workers=16,
        epsilon=1e-8,
        lr=7e-5,
        wdecay=1e-5,
        dropout=0,
        clip=1.0,
        gamma=0.85,
        num_steps=225000,
        val_steps=7500,
        restore_ckpt="ckpts/Tartan-T",
        coarse_config=None
    ):
        self.name = name
        self.dataset = dataset
        self.val_datasets = val_datasets if val_datasets is not None else ["spring", "spring-1080", "sintel", "kitti"]
        self.monitor = monitor
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max
        self.pretrain = pretrain
        self.radius = radius
        self.dim = dim
        self.num_blocks = num_blocks
        self.iters = iters
        self.image_size = image_size if image_size is not None else [864, 1920]
        self.scale = scale
        self.effective_batch_size = effective_batch_size
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.lr = lr
        self.wdecay = wdecay
        self.dropout = dropout
        self.clip = clip
        self.gamma = gamma
        self.num_steps = num_steps
        self.val_steps = val_steps
        self.restore_ckpt = restore_ckpt
        self.coarse_config = coarse_config

class Preprocessor:
    def __init__(self, args, device='cuda'):
        self.video_dir = args.video_dir
        self.output_dir = args.output_dir
        
        self.lmk_extractor = LMKExtractor()
        self.processor = FaceAnimationProcessor(checkpoint=args.smirk_checkpoint)
        self.face_vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False,)
        self.face_helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device=device)
        self.manifest = []

        # Load MEMFOF for optical flow calculation.
        self.memfof_config = MEMFOFConfig()
        self.memfof_model = MEMFOF.from_pretrained(f"egorchistov/optical-flow-{MEMFOF_MODEL}").eval().to(device=device)
        self.args = args
        self.device = device

    def facial_landmarks(self, control_frames):
        """
        Get the facial landmark driving video.
        Args:
            control_frames (torch.Tensor): Tensor of shape (C, T, H, W)
        Returns:
            driving_video (torch.Tensor): shape (C, T, H, W)
        """
        driving_video_crop = []
        ref_faces = []
        # Iterate over the temporal dimension (T) of control_frames
        T = control_frames.shape[1]
        for t in range(T):
            control_frame = control_frames[:, t, :, :]
            frame_np = control_frame.permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            ref_image, x1, y1 = self.processor.face_crop(frame_np)
            ref_faces.append((ref_image, x1, y1))
            driving_video_crop.append(ref_image)

        driving_video = driving_video_crop

        # The original SkyReels A1 inference code separates the first frame and the subsequent frames,
        # but in our self-reenactment approach they are the same. therefore we can populate it more
        # similarly.
        out_frames, landmarks = self.processor.preprocess_lmk3d_self_reenactment(driving_video)
        
        n_frames = len(out_frames)
        c, h, w = control_frames[:, 0, :, :].shape
        input_video = np.zeros((h, w, c), dtype=np.float32)[np.newaxis, :].repeat(n_frames, axis=0)
        for ii in range(n_frames):
            ref_face, x1, y1 = ref_faces[ii]
            face_h, face_w, _ = ref_face.shape
            input_video[ii][y1:y1+face_h, x1:x1+face_w] = out_frames[ii]

            if landmarks[ii] is not None:
                # add (x1, y1) to all items in landmarks
                landmarks[ii][:, 0] += x1
                landmarks[ii][:, 1] += y1

        # concat with remaining motion frames.
        input_video = torch.from_numpy(np.array(input_video))
        input_video = input_video / 255
        # reshape input video to [C, T, H, W]
        input_video = input_video.permute(3, 0, 1, 2)

        return input_video, landmarks
    
    def pixel_mask(self, video_tensor, landmarks):
        """
        Given a video tensor, produce a pixel mask which covers the face region for each frame.
        Args:
            video_tensor (torch.Tensor): Video tensor of shape (C, T, H, W).
            landmarks (list): List of landmarks for each frame, where each item is either None
            or a numpy array of shape (N, 2) containing the (x, y) coordinates of the landmarks.
            None means that mediapipe didn't find any landmarks for that frame.
        Returns:
            torch.Tensor: Pixel mask of shape (1, T, H, W) where 1 indicates inside the face region, 0 otherwise.
        """
        C, T, H, W = video_tensor.shape
        mask = torch.zeros((1, T, H, W), dtype=torch.float32, device=video_tensor.device)
        for i in range(T):
            if landmarks[i] is None:
                # If no landmarks were found, skip this frame. Treat the entire frame as masked
                mask[:, i, :, :] = 1.0
                continue

            image = video_tensor[:, i, :, :].permute(1, 2, 0)  # (H, W, C)
            image = image.cpu().numpy() * 255
            image = image.astype(np.uint8)
            face_mask = np.zeros((H, W), dtype=np.uint8)
            hull = cv2.convexHull(landmarks[i].astype(np.int32))
            cv2.fillPoly(face_mask, [hull], 1)
            # transform face_mask into a torch tensor of shape [1, H, W]
            face_mask = torch.tensor(face_mask, dtype=torch.float32, device=video_tensor.device)
            face_mask = face_mask.unsqueeze(0)  # shape (1, H, W)
            mask[0, i, :, :] = face_mask
        return mask

    @torch.no_grad()
    def optical_flow_mask(self, video_tensor, batch_size=8):
        """
        Given a video tensor, compute the optical flow, using WAFT, on each adjacent pair of frames in batches.
        Args:
            video_tensor (torch.Tensor): Video tensor of shape (C, T, H, W)
            batch_size (int): Number of triplets per batch
        Returns:
            torch.Tensor: Optical flow field of shape (1, T - 2, H, W) 
        """
        _, T, H, W = video_tensor.shape

        # Create as (T - 2, 1, H, W)
        flow_mask = torch.zeros((T - 2, 1, H, W), dtype=torch.float32, device=video_tensor.device)
        triplets = []
        indices = []
        for i in range(T - 2):
            frame1 = video_tensor[:, i, :, :]
            frame2 = video_tensor[:, i + 1, :, :]
            frame3 = video_tensor[:, i + 2, :, :]
            triplet = torch.stack([frame1, frame2, frame3], dim=0)  # (3, 3, H, W)
            triplet = triplet * 255
            triplets.append(triplet)
            indices.append(i)
            if len(triplets) == batch_size or i == T - 3:
                batch = torch.stack(triplets, dim=0)  # (B, 3, 3, H, W)
                output = self.memfof_model(batch.to(device=self.device), fmap_cache=None)
                flows = output['flow'] # (B, 2, 2, H, W)
                # use backward flow
                flows = flows[:, 0]  # (B, 2, H, W)
                # Vectorized flow magnitude calculation
                flow_magnitude = torch.sqrt(flows[:, 0] ** 2 + flows[:, 1] ** 2)  # (B, H, W)
                maximum_magnitude = torch.amax(flow_magnitude, dim=(1, 2), keepdim=True) + 1e-8  # (B, 1, 1)
                # Normalize to [0, 1]
                flow_magnitude = flow_magnitude / maximum_magnitude  # (B, H, W)
                flow_magnitude = flow_magnitude.unsqueeze(1)  # (B, 1, H, W)
                avg_flow = flow_magnitude.mean(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
                high_flow_mask = flow_magnitude >= avg_flow  # (B, 1, H, W)
                # Follow SkyReels A1 section 4.2,
                # We take the top 3/4 of the above-average flow magnitudes.
                # The original paper takes only the top 1/2, but this gave better results anecdotally.
                flow_magnitude = flow_magnitude * high_flow_mask.float() + 0.5
                flow_magnitude[flow_magnitude < 0.75] = 0.0
                # Assign to flow_mask
                flow_mask[indices[0]:indices[-1]+1] = flow_magnitude
                triplets = []
                indices = []
                torch.cuda.empty_cache()
        return flow_mask.float().permute(1, 0, 2, 3)  # Reshape to (1, T-2, H, W)


    def cropped_aligned_identity(self, video_tensor):
        first_frame = video_tensor[:, 0, :, :]
        # convert to numpy
        image = first_frame.permute(1, 2, 0).cpu().numpy() * 255.0
        image = image.astype(np.uint8)

        self.face_helper.clean_all() 
        self.face_helper.read_image(image[:, :, ::-1])
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        align_face = self.face_helper.cropped_faces[0]
        image_face = align_face[:, :, ::-1]

        # convert to pytorch
        image_face = torch.from_numpy(image_face.copy()).permute(2, 0, 1)

        # note: we keep the range in 0, 255
        return image_face

    def preprocess_video(self, video_path):
        """
        Preprocess a single video file.
        This loads the MP4 file, extracts frames, and converts them to a tensor.
        Then it computes facial landmarks, pixel masks, and optical flow masks.
        Then it writes these components to a file in the output directory and updates the manifest.
        Args:
            video_path (str): Path to the video file to preprocess.
        """

        video_id = os.path.splitext(video_path.split("/")[-1])[0]

        video = VideoReader(video_path, num_threads=1)
        frames = video.get_batch(range(len(video))).asnumpy()  # Load all frames as numpy array
        frames_tensor = torch.tensor(frames).permute(3, 0, 1, 2) # Shape is now (C, T, H, W)
        frames_tensor = frames_tensor.float() / 255.0

        _, _, height, width = frames_tensor.shape

        driving_video, landmarks = self.facial_landmarks(frames_tensor)
        pixel_mask = self.pixel_mask(frames_tensor, landmarks)
        optical_flow_mask = self.optical_flow_mask(frames_tensor, batch_size=self.args.optical_batch_size)
        identity_image = self.cropped_aligned_identity(frames_tensor)

        # Save original video. Permute to (T, H, W, C) for video saving
        original_video_np = (frames_tensor.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8) # (T, H, W, C)
        out_path = os.path.join(self.output_dir, f"{video_id}_original.avi")
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(out_path, fourcc, 16, (width, height))
        for frame in original_video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        # Save driving video. Permute to (T, H, W, C) for video saving
        driving_video_np = (driving_video.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8) # (T, H, W, C)
        out_path = os.path.join(self.output_dir, f"{video_id}_driving.avi")
        out = cv2.VideoWriter(out_path, fourcc, 16, (width, height))
        for frame in driving_video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        # Save pixel mask video. Squeeze to (T, H, W) first
        mask_np = (pixel_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8) # (T, H, W)
        out_path = os.path.join(self.output_dir, f"{video_id}_mask.avi")
        out = cv2.VideoWriter(out_path, fourcc, 16, (width, height))
        for frame in mask_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        out.release()

        # Save optical flow mask video. Squeeze to (T, H, W) first
        optical_flow_mask = optical_flow_mask / 1.5 # normalize to [0, 1.0]
        flow_np = (optical_flow_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8) # (T-1, H, W)
        out_path = os.path.join(self.output_dir, f"{video_id}_flow_mask.avi")
        out = cv2.VideoWriter(out_path, fourcc, 16, (width, height))
        for frame in flow_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        out.release()

        # Save identity image (png). Reshape to (H, W, C) and convert to uint8
        identity_image_np = (identity_image.permute(1, 2, 0).cpu().numpy()).astype(np.uint8) # (H, W, C)
        identity_image_path = os.path.join(self.output_dir, f"{video_id}_identity.png")
        cv2.imwrite(identity_image_path, cv2.cvtColor(identity_image_np, cv2.COLOR_RGB2BGR))

        self.manifest.append({
            "video_id": video_id,
            "resolution": f"{width}x{height}",
        })

    def write_manifest(self):
        """
        Write the manifest file to the output directory.
        The manifest contains paths to the preprocessed video files and their metadata.
        """
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=4)
        print(f"Manifest written to {manifest_path}")

def worker_process(args, video_files, worker_id):
    print(f"Worker {worker_id} started with GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    device = "cuda:0"  # Only one GPU is visible to this process
    preprocessor = Preprocessor(args, device=device)
    for video_file in tqdm(video_files, desc=f"Worker {worker_id} processing videos"): 
        video_path = os.path.join(args.video_dir, video_file)
        preprocessor.preprocess_video(video_path)
    # Write worker manifest
    manifest_path = os.path.join(args.output_dir, f"manifest_worker_{worker_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(preprocessor.manifest, f, indent=4)
    print(f"Worker {worker_id} manifest written to {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess training data for SkyReels-A1")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing the video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed data")
    parser.add_argument("--smirk_checkpoint", type=str, required=True, help="Path to the Smirk checkpoint file")
    parser.add_argument("--count", type=int, default=None, help="Maximum number of video files to process")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for preprocessing")
    parser.add_argument("--early_exit", type=bool, default=False, help="Exit early after creating preprocessor")
    parser.add_argument("--optical_batch_size", type=int, default=8, help="Batch size for optical flow mask computation")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
    if args.count is not None:
        video_files = video_files[:args.count]
    if args.num_workers > 1 and not args.early_exit:
        # Split video files into chunks
        chunks = np.array_split(video_files, args.num_workers)
        processes = []
        prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Ensure processes are started with a full spawn.
        multiprocessing.set_start_method('spawn', force=True)
        for worker_id, chunk in enumerate(chunks):
            # set environment for worker
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
            p = multiprocessing.Process(target=worker_process, args=(args, list(chunk), worker_id))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # Restore original CUDA_VISIBLE_DEVICES
        if prev_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
        # Merge manifests
        merged_manifest = []
        for worker_id in range(args.num_workers):
            manifest_path = os.path.join(args.output_dir, f"manifest_worker_{worker_id}.json")
            with open(manifest_path, "r") as f:
                merged_manifest.extend(json.load(f))
            os.remove(manifest_path)
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(merged_manifest, f, indent=4)
        print(f"Final manifest written to {manifest_path}")
    else:
        preprocessor = Preprocessor(args)
        if args.early_exit:
            print("Early exit requested, preprocessor created but no videos processed.")
            return
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(args.video_dir, video_file)
            preprocessor.preprocess_video(video_path)
        preprocessor.write_manifest()


if __name__ == "__main__":
    main()
