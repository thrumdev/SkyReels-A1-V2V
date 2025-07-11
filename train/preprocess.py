import argparse
import os
import json
from decord import VideoReader
import torch
import torch.nn.functional as F
import np
import cv2

from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils  import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from tqdm import tqdm

from .WAFT.model import fetch_model as fetch_waft_model
from .WAFT.inference_tools import InferenceWrapper as WaftInferenceWrapper
from .WAFT.utils.utils import load_ckpt as load_waft_ckpt

# copied from tar-c-t-spring-540p.json
class WaftConfig:
    def __init__(
        self,
        name="tar-c-t-spring-540p",
        dataset="spring",
        gpus=None,
        dav2_backbone="vits",
        network_backbone="vits",
        algorithm="vitwarp",
        use_var=True,
        var_min=0,
        var_max=10,
        iters=5,
        image_size=None,
        scale=-1,
        batch_size=32,
        epsilon=1e-8,
        lr=1e-4,
        wdecay=1e-5,
        dropout=0,
        clip=1.0,
        gamma=0.85,
        num_steps=200000,
        restore_ckpt=None
    ):
        self.name = name
        self.dataset = dataset
        self.gpus = gpus if gpus is not None else [0, 1, 2, 3, 4, 5, 6, 7]
        self.dav2_backbone = dav2_backbone
        self.network_backbone = network_backbone
        self.algorithm = algorithm
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max
        self.iters = iters
        self.image_size = image_size if image_size is not None else [540, 960]
        self.scale = scale
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.lr = lr
        self.wdecay = wdecay
        self.dropout = dropout
        self.clip = clip
        self.gamma = gamma
        self.num_steps = num_steps
        self.restore_ckpt = restore_ckpt

waft_config = WaftConfig()

class Preprocessor:
    def __init__(self, args):
        self.video_dir = args.video_dir
        self.output_dir = args.output_dir
        
        self.lmk_extractor = LMKExtractor()
        self.processor = FaceAnimationProcessor(checkpoint=args.smirk_checkpoint)
        self.face_vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False,)
        self.face_helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device="cuda",)
        self.manifest = []

        # Load WAFT for optical flow
        self.waft_config = waft_config
        self.waft_model = fetch_waft_model(waft_config)
        load_waft_ckpt(self.waft_model, args.waft_checkpoint)
        self.waft_model.cuda()
        self.waft_model.eval()
        self.waft_model = WaftInferenceWrapper(
            self.waft_model, 
            scale=waft_config.scale, 
            train_size=waft_config.image_size,
            pad_to_train_size=False,
            tiling=False,
        )
        self.args = args

    def facial_landmarks(self, control_frames):
        driving_video_crop = []
        ref_faces = []
        for control_frame in control_frames:
            ref_image, x1, y1 = self.processor.face_crop(np.array(control_frame))
            ref_faces.append((ref_image, x1, y1))
            driving_video_crop.append(ref_image)

        driving_video = driving_video_crop

        # The original SkyReels A1 inference code separates the first frame and the subsequent frames,
        # but in our self-reenactment approach they are the same. therefore we can populate it more
        # similarly.
        out_frames = self.processor.preprocess_lmk3d_multi(driving_video, driving_video)
        
        input_video = np.zeros(control_frames[0].shape, dtype=np.float32)[np.newaxis, :].repeat(49, axis=0)
        for ii in range(49):
            ref_face, x1, y1 = ref_faces[ii]
            face_h, face_w, _ = ref_face.shape
            input_video[ii][y1:y1+face_h, x1:x1+face_w] = out_frames[ii]

        # concat with remaining motion frames.
        input_video = torch.from_numpy(np.array(input_video))
        input_video = input_video / 255
        # reshape input video to [C, T, H, W]
        input_video = input_video.permute(1, 0, 2, 3)

        return input_video
    
    def pixel_mask(self, video_tensor):
        """
        Given a video tensor, produce a pixel mask which covers the face region for each frame.
        Args:
            video_tensor (torch.Tensor): Video tensor of shape (C, T, H, W).
        Returns:
            torch.Tensor: Pixel mask of shape (1, T, H, W) where 1 indicates inside the face region, 0 otherwise.
        """
        C, T, H, W = video_tensor.shape
        mask = torch.zeros((1, T, H, W), dtype=torch.float32, device=video_tensor.device)
        for i in range(T):
            face_mask = self.processor.face_mask(video_tensor[i])
            mask[0, i, :, :] = face_mask
        return mask

    def optical_flow_mask(self, video_tensor):
        """
        Given a video tensor, compute the optical flow, using WAFT, on each adjacent pair of frames
        Args:
            video_tensor (torch.Tensor): Video tensor of shape (C, T, H, W)
        Returns:
            torch.Tensor: Optical flow field of shape (1, T - 1, H, W) 
        """

        _, T, H, W = video_tensor.shape
        flow_mask = torch.zeros((1, T - 1, H, W), dtype=torch.float32, device=video_tensor.device)
        for i in range(video_tensor.shape[1] - 1):
            frame1 = video_tensor[:, i, :, :] # (C, H, W)
            frame2 = video_tensor[:, i + 1, :, :]

            # Scale frame1 and frame2 to the waft config desired image size.
            frame1 = F.interpolate(
                frame1.unsqueeze(0), 
                size=self.waft_config.image_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            frame2 = F.interpolate(
                frame2.unsqueeze(0), 
                size=self.waft_config.image_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

            output = self.waft_model.calc_flow(frame1, frame2)

            # Output contains `waft_config.iters` items. last is best.
            flow = output['flow'][-1][0] # shape (2, H, W)

            # Convert flow to optical flow mask.
            flow_magnitude = torch.sqrt(flow[0] ** 2 + flow[1] ** 2)  # shape (H, W)
            maximum_magnitude = flow_magnitude.max().item() + 1e-8

            # Normalize the flow to [0, 1]
            flow_magnitude = flow_magnitude / maximum_magnitude
            flow_magnitude = flow_magnitude.unsqueeze(0)  # shape (1, H, W)

            # Scale back to original size.
            flow_magnitude = F.interpolate(flow_magnitude, size=(H, W), mode='bilinear', align_corners=False)

            avg_flow = flow_magnitude.mean().item()
            high_flow_mask = flow_magnitude >= avg_flow

            flow_magnitude = flow_magnitude * high_flow_mask.float() + 0.5
            # set all items < 1.0 to 0 and clamp to 1.5
            # this is the M_i,norm described in section 4.2 of the skyreels A1 paper.
            # shape (1, H, W)
            flow_magnitude[flow_magnitude < 1.0] = 0.0
            flow_magnitude[flow_magnitude > 1.5] = 1.5

            flow_mask[:, i, :, :] = flow_magnitude

        return flow_mask.float()


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
        image_face = torch.from_numpy(image_face).permute(2, 0, 1)

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

        video_filename = video_path.split("/")[-1]
        output_filename = f"{video_filename}.pt"
        output_file = os.path.join(self.output_dir, output_filename)

        video = VideoReader(video_path, num_threads=1)
        frames = video.get_batch(range(len(video))).asnumpy()  # Load all frames as numpy array
        frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2) # Shape is now (C, T, H, W)
        frames_tensor = frames_tensor.float() / 255.0

        _, _, height, width = frames_tensor.shape

        driving_video = self.facial_landmarks(frames_tensor)
        pixel_mask = self.pixel_mask(frames_tensor)
        optical_flow_mask = self.optical_flow_mask(frames_tensor)

        with open(output_file, "wb") as f:
            torch.save({
                "ref_video": frames_tensor,
                "driving_video": driving_video,
                "mask": pixel_mask,
                "optical_flow_mask": optical_flow_mask,
                "cropped_aligned_identity": self.cropped_aligned_identity(frames_tensor),
            }, f)

        # Save processed videos if requested
        if getattr(self.args, 'save_videos', False):
            # Save driving video
            driving_video_np = (driving_video.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8) # (T, H, W, C)
            out_path = os.path.join(self.output_dir, f"{video_filename}_driving.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, 25, (width, height))
            for frame in driving_video_np:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

            # Save pixel mask video
            mask_np = (pixel_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) # (T, H, W)
            out_path = os.path.join(self.output_dir, f"{video_filename}_mask.mp4")
            out = cv2.VideoWriter(out_path, fourcc, 25, (width, height))
            for frame in mask_np:
                out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            out.release()

            # Save optical flow mask video
            flow_np = (optical_flow_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) # (T-1, H, W)
            out_path = os.path.join(self.output_dir, f"{video_filename}_flow_mask.mp4")
            out = cv2.VideoWriter(out_path, fourcc, 25, (width, height))
            for frame in flow_np:
                out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            out.release()

        self.manifest.append({
            "filepath": output_filename,
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

def main():
    parser = argparse.ArgumentParser(description="Preprocess training data for SkyReels-A1")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing the video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed data")
    parser.add_argument("--waft_checkpoint", type=str, required=True, help="Path to the WAFT checkpoint file")
    parser.add_argument("--smirk_checkpoint", type=str, required=True, help="Path to the Smirk checkpoint file")
    parser.add_argument("--save_videos", action="store_true", help="If set, save video files as well as raw tensors")
    parser.add_argument("--count", type=int, default=None, help="Maximum number of video files to process")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocessor = Preprocessor(args)
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
    if args.count is not None:
        video_files = video_files[:args.count]
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(args.video_dir, video_file)
        preprocessor.preprocess_video(video_path)

    preprocessor.write_manifest()


if __name__ == "__main__":
    main()
