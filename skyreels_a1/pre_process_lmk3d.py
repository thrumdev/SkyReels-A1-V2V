import torch
import cv2
import os
import sys
import numpy as np
import argparse
import math
from PIL import Image
from decord import VideoReader
from skimage.transform import estimate_transform, warp
from insightface.app import FaceAnalysis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skyreels_a1.src.utils.mediapipe_utils import MediaPipeUtils
from skyreels_a1.src.smirk_encoder import SmirkEncoder
from skyreels_a1.src.FLAME.FLAME import FLAME
from skyreels_a1.src.renderer import Renderer

class FaceAnimationProcessor:
    def __init__(self, device='cuda', checkpoint="pretrained_models/smirk/smirk_encoder.pt"):
        self.device = device
        self.app = FaceAnalysis(allowed_modules=['detection'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.smirk_encoder = SmirkEncoder().to(device)
        self.flame = FLAME(n_shape=300, n_exp=50).to(device)
        self.renderer = Renderer().to(device)
        self.mediapipe_utils = MediaPipeUtils()
        self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        checkpoint_data = torch.load(checkpoint)
        checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint_data.items() if 'smirk_encoder' in k}
        self.smirk_encoder.load_state_dict(checkpoint_encoder)
        self.smirk_encoder.eval().cuda()

    def face_mask(self, image):
        """
        Yields a binary mask of the same H, W as `image`, using insightface.app.FaceAnalysis
        to select only the pixels included in the landmarks.

        Args:
            image: torch.Tensor (H, W, C)
        """

        image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        faces = self.app.get(image)
        kps = faces[0]['kps']
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array(kps, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 1)
        return mask
    
    def faces(self, image):
        faces = self.app.get(image)
        return faces

    def face_crop(self, image):
        height, width, _ = image.shape
        faces = self.app.get(image)
        bbox = faces[0]['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x1 = max(0, int(bbox[0] - w/2))
        x2 = min(width - 1, int(bbox[2] + w/2))
        w_new = x2 - x1
        y_offset = (w_new - h) / 2.
        y1 = max(0, int(bbox[1] - y_offset))
        y2 = min(height, int(bbox[3] + y_offset))
        x_comp = int(((x2 - x1) - (y2 - y1)) / 2) if (x2 - x1) > (y2 - y1) else 0
        x1 += x_comp
        x2 -= x_comp
        image_crop = image[y1:y2, x1:x2]
        return image_crop, x1, y1

    def face_crop_with_provided(self, image, face):
        height, width, _ = image.shape
        bbox = face['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x1 = max(0, int(bbox[0] - w/2))
        x2 = min(width - 1, int(bbox[2] + w/2))
        w_new = x2 - x1
        y_offset = (w_new - h) / 2.
        y1 = max(0, int(bbox[1] - y_offset))
        y2 = min(height, int(bbox[3] + y_offset))
        x_comp = int(((x2 - x1) - (y2 - y1)) / 2) if (x2 - x1) > (y2 - y1) else 0
        x1 += x_comp
        x2 -= x_comp
        image_crop = image[y1:y2, x1:x2]
        return image_crop, x1, y1

    def crop_and_resize(self, image, height, width):
        image = np.array(image)
        image_height, image_width, _ = image.shape
        if image_height / image_width < height / width:
            croped_width = int(image_height / height * width)
            left = (image_width - croped_width) // 2
            image = image[:, left: left+croped_width]
        else:
            pad = int((((width / height) * image_height) - image_width) / 2.)
            padded_image = np.zeros((image_height, image_width + pad * 2, 3), dtype=np.uint8)
            padded_image[:, pad:pad+image_width] = image
            image = padded_image
        return Image.fromarray(image).resize((width, height))

    def rodrigues_to_matrix(self, pose_params):
        theta = torch.norm(pose_params, dim=-1, keepdim=True)
        r = pose_params / (theta + 1e-8)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        r_x = torch.zeros((pose_params.shape[0], 3, 3), device=pose_params.device)
        r_x[:, 0, 1] = -r[:, 2]
        r_x[:, 0, 2] = r[:, 1]
        r_x[:, 1, 0] = r[:, 2]
        r_x[:, 1, 2] = -r[:, 0]
        r_x[:, 2, 0] = -r[:, 1]
        r_x[:, 2, 1] = r[:, 0]
        R = cos_theta * torch.eye(3, device=pose_params.device).unsqueeze(0) + \
            sin_theta * r_x + \
            (1 - cos_theta) * r.unsqueeze(-1) @ r.unsqueeze(-2)
        return R

    def matrix_to_rodrigues(self, R):
        cos_theta = (torch.trace(R[0]) - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        theta = torch.acos(cos_theta)
        if abs(theta) < 1e-4:
            return torch.zeros(1, 3, device=R.device)
        elif abs(theta - math.pi) < 1e-4:
            R_plus_I = R[0] + torch.eye(3, device=R.device)
            col_norms = torch.norm(R_plus_I, dim=0)
            max_col_idx = torch.argmax(col_norms)
            v = R_plus_I[:, max_col_idx]
            v = v / torch.norm(v)
            return (v * math.pi).unsqueeze(0)
        sin_theta = torch.sin(theta)
        r = torch.zeros(1, 3, device=R.device)
        r[0, 0] = R[0, 2, 1] - R[0, 1, 2]
        r[0, 1] = R[0, 0, 2] - R[0, 2, 0]
        r[0, 2] = R[0, 1, 0] - R[0, 0, 1]
        r = r / (2 * sin_theta)
        return r * theta

    def crop_face(self, frame, landmarks, scale=1.0, image_size=224):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])
        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * scale)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        tform_original = estimate_transform('similarity', src_pts, src_pts)
        return tform, tform_original

    def compute_landmark_relation(self, kpt_mediapipe, target_idx=473, ref_indices=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]):
        target_point = kpt_mediapipe[target_idx]
        ref_points = kpt_mediapipe[ref_indices]
        left_corner = ref_points[0]
        right_corner = ref_points[8]
        eye_center = (left_corner + right_corner) / 2
        eye_width_vector = right_corner - left_corner
        eye_width = np.linalg.norm(eye_width_vector)
        eye_direction = eye_width_vector / eye_width
        eye_vertical = np.array([-eye_direction[1], eye_direction[0]])
        target_vector = target_point - eye_center
        x_relative = np.dot(target_vector, eye_direction) / (eye_width/2)
        y_relative = np.dot(target_vector, eye_vertical) / (eye_width/2)
        return [np.array([x_relative, y_relative]),target_point,ref_points,ref_indices]

    def process_source_image(self, image_rgb, input_size=224):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        kpt_mediapipe, _, _, mediapipe_eye_pose = self.mediapipe_utils.run_mediapipe(image_bgr)
        if kpt_mediapipe is None:
            raise ValueError('Cannot find facial landmarks in the source image')
        kpt_mediapipe = kpt_mediapipe[..., :2]
        tform, _ = self.crop_face(image_rgb, kpt_mediapipe, scale=1.4, image_size=input_size)
        cropped_image = warp(image_rgb, tform.inverse, output_shape=(input_size, input_size), preserve_range=True).astype(np.uint8)
        cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cropped_image = cropped_image.to(self.device)
        with torch.no_grad():
            source_outputs = self.smirk_encoder(cropped_image)
        source_outputs['eye_pose_params'] = torch.tensor(mediapipe_eye_pose).to(self.device)
        return source_outputs, tform, image_rgb

    def smooth_params(self, data, alpha=0.7):
        smoothed_data = [data[0]]
        for i in range(1, len(data)):
            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
            smoothed_data.append(smoothed_value)
        return smoothed_data

    def process_driving_img_list(self, img_list, input_size=224):
        driving_frames = []
        driving_outputs = []
        driving_tforms = []
        original_images = []
        weights_473 = []
        weights_468 = []
        mediapipe_landmarks = []
        for i, frame in enumerate(img_list):
            img_rgb = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            try:
                kpt_mediapipe, mediapipe_exp, mediapipe_pose, mediapipe_eye_pose = self.mediapipe_utils.run_mediapipe(frame)
            except:
                print('Warning: No face detected in a frame, skipping this frame')
                driving_frames.append(None)
                driving_outputs.append(None)
                driving_tforms.append(None)
                original_images.append(img_rgb)
                weights_473.append(None)
                weights_468.append(None)
                mediapipe_landmarks.append(None)
                continue
            if kpt_mediapipe is None:
                print('Warning: No face detected in a frame, skipping this frame')
                driving_frames.append(None)
                driving_outputs.append(None)
                driving_tforms.append(None)
                original_images.append(img_rgb)
                weights_473.append(None)
                weights_468.append(None)
                mediapipe_landmarks.append(None)
                continue

            kpt_mediapipe = kpt_mediapipe[..., :2]
            mediapipe_landmarks.append(kpt_mediapipe)
            weights_473.append(self.compute_landmark_relation(kpt_mediapipe))
            weights_468.append(self.compute_landmark_relation(kpt_mediapipe, target_idx=468, ref_indices=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]))
            
            tform, _ = self.crop_face(frame, kpt_mediapipe, scale=1.4, image_size=input_size)
            cropped_frame = warp(frame, tform.inverse, output_shape=(input_size, input_size), preserve_range=True).astype(np.uint8)
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            cropped_frame = torch.tensor(cropped_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            cropped_frame = cropped_frame.to(self.device)
            with torch.no_grad():
                outputs = self.smirk_encoder(cropped_frame)
            outputs['eye_pose_params'] = torch.tensor(mediapipe_eye_pose).to(self.device)
            outputs['mediapipe_exp'] = torch.tensor(mediapipe_exp).to(self.device)
            outputs['mediapipe_pose'] = torch.tensor(mediapipe_pose).to(self.device)
            driving_frames.append(frame)
            driving_outputs.append(outputs)
            driving_tforms.append(tform)
            original_images.append(img_rgb)
        return driving_frames, driving_outputs, driving_tforms, weights_473, weights_468, original_images, mediapipe_landmarks

    def preprocess_lmk3d(self, source_image=None, driving_image_list=None):
        source_outputs, source_tform, image_original = self.process_source_image(source_image)
        _, driving_outputs, driving_video_tform, weights_473, weights_468, _, _ = self.process_driving_img_list(driving_image_list)
        driving_outputs_list = []
        source_pose_init = source_outputs['pose_params'].clone()
        driving_outputs_pose = [outputs['pose_params'] for outputs in driving_outputs]
        driving_outputs_pose = self.smooth_params(driving_outputs_pose)
        for i, outputs in enumerate(driving_outputs):
            outputs['pose_params'] = driving_outputs_pose[i]
            source_outputs['expression_params'] = outputs['expression_params']
            source_outputs['jaw_params'] = outputs['jaw_params']
            source_outputs['eye_pose_params'] = outputs['eye_pose_params']
            source_matrix = self.rodrigues_to_matrix(source_pose_init)
            driving_matrix_0 = self.rodrigues_to_matrix(driving_outputs[0]['pose_params'])
            driving_matrix_i = self.rodrigues_to_matrix(driving_outputs[i]['pose_params'])
            relative_rotation = torch.inverse(driving_matrix_0) @ driving_matrix_i
            new_rotation = source_matrix @ relative_rotation
            source_outputs['pose_params'] = self.matrix_to_rodrigues(new_rotation)
            source_outputs['eyelid_params'] = outputs['eyelid_params']
            flame_output = self.flame.forward(source_outputs)
            renderer_output = self.renderer.forward(
                flame_output['vertices'],
                source_outputs['cam'],
                landmarks_fan=flame_output['landmarks_fan'], source_tform=source_tform,
                tform_512=None, weights_468=weights_468[i], weights_473=weights_473[i],
                landmarks_mp=flame_output['landmarks_mp'], shape=image_original.shape)
            rendered_img = renderer_output['rendered_img']
            driving_outputs_list.extend(np.copy(rendered_img)[np.newaxis, :])
        return driving_outputs_list
    
        
    def preprocess_lmk3d_self_reenactment(self, driving_image_list):
        _, driving_outputs, source_tform, weights_473, weights_468, image_original, mediapipe_landmarks = self.process_driving_img_list(driving_image_list)
        driving_outputs_list = []
        for i, outputs in enumerate(driving_outputs):
            if outputs is None or image_original[i] is None:
                print(f"Skipping frame {i} (writing black frame)")
                black_frame = np.zeros_like(image_original[i])
                driving_outputs_list.extend(black_frame[np.newaxis, :])
                continue

            # Optionally smooth pose
            flame_output = self.flame.forward(outputs)
            renderer_output = self.renderer.forward(
                flame_output['vertices'],
                outputs['cam'],
                landmarks_fan=flame_output['landmarks_fan'], source_tform=source_tform[i],
                tform_512=None, weights_468=weights_468[i], weights_473=weights_473[i],
                landmarks_mp=flame_output['landmarks_mp'], shape=image_original[i].shape)
            rendered_img = renderer_output['rendered_img']
            driving_outputs_list.extend(np.copy(rendered_img)[np.newaxis, :])
        return driving_outputs_list, mediapipe_landmarks

    def preprocess_lmk3d_from_coef(self, source_outputs, source_tform, render_shape, driving_outputs):
        driving_outputs_list = []
        source_pose_init = source_outputs['pose_params'].clone()
        driving_outputs_pose = [outputs['pose_params'] for outputs in driving_outputs]
        driving_outputs_pose = self.smooth_params(driving_outputs_pose)
        for i, outputs in enumerate(driving_outputs):
            outputs['pose_params'] = driving_outputs_pose[i]
            source_outputs['expression_params'] = outputs['expression_params']
            source_outputs['jaw_params'] = outputs['jaw_params']
            source_outputs['eye_pose_params'] = outputs['eye_pose_params']
            source_matrix = self.rodrigues_to_matrix(source_pose_init)
            driving_matrix_0 = self.rodrigues_to_matrix(driving_outputs[0]['pose_params'])
            driving_matrix_i = self.rodrigues_to_matrix(driving_outputs[i]['pose_params'])
            relative_rotation = torch.inverse(driving_matrix_0) @ driving_matrix_i
            new_rotation = source_matrix @ relative_rotation
            source_outputs['pose_params'] = self.matrix_to_rodrigues(new_rotation)
            source_outputs['eyelid_params'] = outputs['eyelid_params']
            flame_output = self.flame.forward(source_outputs)
            renderer_output = self.renderer.forward(
                flame_output['vertices'],
                source_outputs['cam'],
                landmarks_fan=flame_output['landmarks_fan'], source_tform=source_tform,
                tform_512=None, weights_468=None, weights_473=None,
                landmarks_mp=flame_output['landmarks_mp'], shape=render_shape)
            rendered_img = renderer_output['rendered_img']
            driving_outputs_list.extend(np.copy(rendered_img)[np.newaxis, :])
        return driving_outputs_list


    def ensure_even_dimensions(self, frame):
        height, width = frame.shape[:2]
        new_width = width - (width % 2)
        new_height = height - (height % 2)
        if new_width != width or new_height != height:
            frame = cv2.resize(frame, (new_width, new_height))
        return frame

    def get_global_bbox(self, frames):
        max_x1, max_y1, max_x2, max_y2 = float('inf'), float('inf'), 0, 0
        for frame in frames:
            faces = self.app.get(frame)
            if not faces:
                continue
            bbox = faces[0]['bbox']
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            max_x1 = min(max_x1, x1)
            max_y1 = min(max_y1, y1)
            max_x2 = max(max_x2, x2)
            max_y2 = max(max_y2, y2)
        w = max_x2 - max_x1
        h = max_y2 - max_y1
        x1 = int(max_x1 - w / 2)
        x2 = int(max_x2 + w / 2)
        y_offset = (x2 - x1 - h) / 2
        y1 = int(max_y1 - y_offset)
        y2 = int(max_y2 + y_offset)
        if (x2 - x1) > (y2 - y1):
            x_comp = int(((x2 - x1) - (y2 - y1)) / 2)
            x1 += x_comp
            x2 -= x_comp
        else:
            y_comp = int(((y2 - y1) - (x2 - x1)) / 2)
            y1 += y_comp
            y2 -= y_comp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frames[0].shape[1], x2)
        y2 = min(frames[0].shape[0], y2)
        return int(x1), int(y1), int(x2), int(y2)

    def face_crop_with_global_box(self, image, global_box):
        x1, y1, x2, y2 = global_box
        return image[y1:y2, x1:x2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and image for face animation.")
    parser.add_argument('--source_image', type=str, default="assets/ref_images/1.png", help='Path to the source image.')
    parser.add_argument('--driving_video', type=str, default="assets/driving_video/1.mp4", help='Path to the driving video.')
    parser.add_argument('--output_path', type=str, default="./output.mp4", help='Path to save the output video.')
    args = parser.parse_args()

    processor = FaceAnimationProcessor(checkpoint='pretrained_models/smirk/SMIRK_em1.pt')
    processor.process_video(args.source_image, args.driving_video, args.output_path)
