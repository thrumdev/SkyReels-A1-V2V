import torch
import torch.nn.functional as F

from diffusers.models import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.schedulers import CogVideoXDDIMScheduler
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from einops import rearrange

from typing import Tuple

from ..skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

class SkyReelsA1V2VInpaintPipeline:
    def __init__(self, config, device):
        self.config = config
        model_name = config.get("model_path", "pretrained_models/SkyReels-A1-5B")
        siglip_name = config.get("siglip_path", "pretrained_models/SkyReels-A1-5B/siglip-so400m-patch14-384")
        self.dtype = config.get("dtype", "bfloat16")
        self.device = device

        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
        ).to(self.dtype, device)

        self.transformer.expand_proj_channels(48 + 64) # Add the mask channels if necessary.

        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(self.dtype, device)

        self.lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="pose_guider",
        ).to(self.dtype, device)

        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler"
        )

        self.siglip = SiglipVisionModel.from_pretrained(siglip_name).to(self.dtype, device)
        self.siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_name).to(self.dtype, device)

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings
    @torch.no_grad()
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.transformer.config.patch_size
        vae_scale_factor_spatial = 8
        grid_height = height // (vae_scale_factor_spatial * p)
        grid_width = width // (vae_scale_factor_spatial * p)
        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def prepare_masks(self, pixel_masks):
        """
            Given a list of pixel masks, this function prepares masks in latent space.
            These have 64 channels, to account for the 8x8 downsampling of the VAE.
        """
        result_masks = []
        for mask in pixel_masks:
            mask = mask.to(torch.bfloat16)
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // 4)
            height = 2 * (int(height) // (8 * 2))
            width = 2 * (int(width) // (8 * 2))

            # scale by 8x8 to match the latent space of the VAE
            mask = rearrange(
                mask,
                "1 t (h ph) (w pw) -> (ph pw) t h w",
                ph=8,
                pw=8,
            )

            # temporal interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            result_masks.append(mask)

        return result_masks
    
    @torch.no_grad()
    def prepare_latent(self, ref_videos, driving_videos, pixel_masks, latent_masks, timesteps):
        """
        Prepares the latent representations of the reference and driving videos, as well as the pixel masks.

        Args:
            ref_videos: List of reference video tensors (C, T, H, W)
            driving_videos: List of driving video tensors (C, T, H, W)
            pixel_masks: List of pixel mask tensors (1, T, H, W)
            latent_masks: List of latent mask tensors (64, T', H', W')
            timesteps: List of timesteps

        Returns model inputs (B, C, T', H', W') and noise (B, C, T', H', W').
        The model inputs are concatenated along the channel dimension in the order:
            [noisy_latent, lmk_latent, ref_latent, latent_mask]
        """

        all_noise = []
        model_inputs = []

        # ignore the first frame of the pixel mask.
        for ref_video, driving_video, pixel_mask, latent_mask, timestep, in zip(ref_videos, driving_videos, pixel_masks, latent_masks, timesteps):
            ref_video = ref_video.to(self.device)
            driving_video = driving_video.to(self.device)
            pixel_mask = pixel_mask.to(self.device)
            latent_mask = latent_mask.to(self.device)
            
            clean_latent = self.vae.encode(ref_video).latent_dist.sample()
            noise = torch.randn_like(clean_latent, device=ref_video.device, dtype=self.dtype)
            noisy_latent = self.scheduler.add_noise(
                clean_latent, 
                noise, 
                torch.tensor([timestep], dtype=torch.int64, device=self.device)
            )
            noisy_latent = self.scheduler.scale_model_input(noisy_latent, timestep)
            
            # Mask the reference video by the pixel mask, except the first frame.
            pixel_mask = pixel_mask[:, 1:, :, :]
            ref_video[:, 1:, : :] *= pixel_mask
            ref_latent = self.vae.encode(ref_video)[0].mode()

            lmk_latent = self.lmk_encoder.encode(driving_video)[0].mode()
            lmk_latent = lmk_latent * self.lmk_encoder.config.scaling_factor

            # concatenate along channel dimension.
            model_input = torch.cat([noisy_latent, lmk_latent, ref_latent, latent_mask], dim=0)
            model_inputs.append(model_input)
            all_noise.append(noise)

        # Stack all model inputs and noise tensors along a freshly introduced batch dimension.
        model_inputs = torch.stack(model_inputs, dim=0)
        all_noise = torch.stack(all_noise, dim=0)

        return model_inputs, all_noise

    @torch.no_grad()
    def embed_reference_prompt(self, identity_images):
        """
        Embeds each identity image using the Siglip model.

        Args:
            identity_images: List of reference video images (C, H, W) with values in [0, 255]

        Returns:
            A tensor of shape [B, 729, 1152] where B is the number of reference videos.
        """
        
        all_image_embeddings = []
        for identity_image in identity_images:
            image = identity_image.to(self.dtype, self.device)
            imgs = self.siglip_normalize.preprocess(images=[image], do_resize=True, return_tensors="pt", do_convert_rgb=True)
            image_embeddings = self.siglip(**imgs.to(dtype=self.dtype)).last_hidden_state # torch.Size([1, 729, 1152])
            all_image_embeddings.append(image_embeddings)

        return torch.cat(all_image_embeddings, dim=0).to(self.device)


    def step_forward_and_loss(
        self, 
        ref_videos, 
        driving_videos, 
        pixel_masks,
        optical_flow_masks,
        identity_images,
        timesteps,
        height,
        width,
        frames
    ):
        """
        Performs a forward pass of the SkyReels A1 V2V inpainting pipeline. Returns the loss.

        Args:
            ref_videos: List of reference video tensors (C, T, H, W)
            driving_videos: List of driving video tensors (C, T, H, W)
            pixel_masks: List of pixel mask tensors (1, T, H, W)
            optical_flow_masks: List of optical flow mask tensors (1, T-2, H, W)
            identity_images: List of identity images (H, W)
            timesteps: List of timesteps
            height: Height of the input videos
            width: Width of the input videos
            frames: Number of frames in the input videos
        """

        latent_masks = self.prepare_masks(pixel_masks)
        model_inputs, noise_gt = self.prepare_latent(
            ref_videos, 
            driving_videos, 
            pixel_masks, 
            latent_masks, 
            timesteps
        )
        image_embeddings = self.embed_reference_prompt(identity_images)
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height=height,
            width=width,
            num_frames=frames,
            device=self.device,
        )

        noise_pred = self.transformer(
            hidden_states=model_inputs.to(self.transformer.device, self.dtype),
            encoder_hidden_states=image_embeddings.to(self.transformer.device, self.dtype),
            timestep=torch.stack(timesteps, dim=0).to(self.transformer.device, self.dtype),
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()

        latent_masks = self.flatten_latent_mask(latent_masks).to(self.device)
        inpaint_loss = self.compute_inpaint_loss(noise_pred, noise_gt, latent_masks)
        optical_flow_loss = self.compute_masked_optical_flow_loss(
            noise_pred,
            noise_gt,
            latent_masks,
            optical_flow_masks,
        )
        
        return self.config.get("inpaint_lambda", 1.0) * inpaint_loss + self.config.get("optical_flow_lambda", 1.0) * optical_flow_loss
    

    @torch.no_grad()
    def flatten_latent_mask(self, latent_masks):
        """
        Flattens the latent mask from (B, 64, T', H', W') to (B, 1, T', H', W').
        This is done by taking the mean across the channel dimension.

        Args:
            latent_masks: A list of B latent mask tensors, each of shape (64, T', H', W').
        
        Returns:
            A single tensor of shape (B, 1, T', H', W') where each channel has been averaged.
        """

        latent_mask = torch.stack(latent_masks, dim=0)
        return latent_mask.mean(dim=1, keep_dim=True)
    
    def compute_inpaint_loss(self, noise_pred, noise_gt, latent_mask):
        """
        Computes the MSE loss between noise_pred and noise_gt, but only for elements where latent_mask == 0 (i.e., outside the mask).
        Args:
            noise_pred: Predicted noise tensor (B, C, T, H, W) or (B, C, H, W)
            noise_gt: Ground truth noise tensor, same shape as noise_pred
            latent_mask: Binary mask tensor, (B, 1, T', H', W') where 1=inside mask, 0=outside.
                intermediate values indicate partial masking within the latent subpixel.
        Returns:
            Scalar loss (averaged over outside-mask elements)
        """
        outside_mask = (1.0 - latent_mask).float()
        loss = ((noise_pred - noise_gt) ** 2) * outside_mask
        denom = outside_mask.sum() + 1e-8
        return loss.sum() / denom

    def compute_masked_optical_flow_loss(self, noise_pred, noise_gt, latent_mask, optical_flow_mask):
        """
        Computes the masked optical flow loss between ref_video and gen_video, using pixel_mask 
        to filter out masked regions entirely.

        Args:
            noise_pred: Predicted noise tensor (B, C, T', H', W')
            noise_gt: Ground truth noise tensor, same shape as noise_pred
            latent_mask: Binary mask tensor, (B, 1, T', H', W') where 1=inside mask, 0=outside.
                intermediate values indicate partial masking within the latent subpixel.
            optical_flow_mask: List of optical B flow mask tensors for the reference video (1, T-2, H, W)
                these contain values either of [1.0, 1.5] or [0].
            
        Note that H' and W' are the latent dimensions, which are smaller than the original dimensions
        H and W due to downsampling in the VAE.

        Following the SkyReels-A1 paper, this loss is computed by:
        1. We first downsample the optical flow mask to (B, 1, T', H', W')
        2. We merge the latent_mask and optical_flow_mask by multiplying to create a combined mask M.
        3. We then compute the total MSE loss between noise_pred and noise_gt, scaled by 
              the combined mask M, and averaged over all latent pixels.

        We average over the _total_ number of pixels, following the definition of the face-aware
        loss in the SkyReels-A1 paper (section 4.2) with a few modifications.

        Steps 1 and 2 are no_grad.
        """

        with torch.no_grad():
            target_shape = noise_pred.shape[-3:]
            optical_flow_mask = torch.stack(optical_flow_mask, dim=0)  # (B, 1, T-2, H, W)
            mask = torch.nn.functional.interpolate(
                optical_flow_mask.float(),
                size=target_shape,
                mode='trilinear',
                align_corners=False
            )
            # set all values in mask to at least 0.1.
            # this is again a divergence from skyreels-a1, but i believe we will want at least
            # some training signal to reach the 'low-motion' face pixels.
            mask = torch.clamp(mask, min=0.1)
            combined_mask = mask * latent_mask.float()

        loss = ((noise_pred.float() - noise_gt.float()) ** 2) * combined_mask
        denom = loss.numel()  # average over all latent pixels, as described in the doc comment
        return loss.sum() / denom
