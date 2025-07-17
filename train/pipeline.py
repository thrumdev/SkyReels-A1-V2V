import torch
import torch.nn.functional as F

from diffusers.models import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.schedulers import CogVideoXDDIMScheduler
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from einops import rearrange
from tqdm import tqdm

from typing import Tuple
import inspect

from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel

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

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps_for_inference(
    scheduler,
    num_inference_steps,
    device,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SkyReelsA1V2VInpaintPipeline:
    def __init__(self, config, device):
        self.config = config
        model_name = config.get("model_path", "pretrained_models/SkyReels-A1-5B")
        siglip_name = config.get("siglip_path", "pretrained_models/SkyReels-A1-5B/siglip-so400m-patch14-384")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.device = device

        restore_checkpoint = config.get("restore_checkpoint", None)

        # We restore the full model only if no lora rank is specified.
        if restore_checkpoint is None or config.get("lora_rank") is not None:
            self.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_name,
                subfolder="transformer",
            ).to(device, self.dtype)
        else:
            print("Restoring full model")

            self.transformer = CogVideoXTransformer3DModel.from_pretrained(
                restore_checkpoint,
            ).to(device, self.dtype)

        self.explicit_mask_channels = self.config.get("explicit_mask_channels", False)
        self.ref_frames_strength = self.config.get("ref_frames_strength", 0.01)
        self.ref_frames_strength = min(0.0, max(1.0, self.ref_frames_strength))

        if self.explicit_mask_channels:
            self.transformer.patch_embed.expand_proj_channels(48 + 64) # Add the mask channels if necessary.

        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(device, self.dtype)
        self.vae.enable_tiling()

        self.lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="pose_guider",
        ).to(device, self.dtype)

        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler"
        )
        self.inference_scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler"
        )
        self.inference_timesteps = 20
        self.vae_scaling_factor_image = self.vae.config.scaling_factor
        self.lmk_scaling_factor_image = self.lmk_encoder.config.scaling_factor

        if config.get("compile", False):
            self.vae.encode = torch.compile(self.vae.encode)
            self.lmk_encoder.encode = torch.compile(self.lmk_encoder.encode)

        self.siglip = SiglipVisionModel.from_pretrained(siglip_name).to(device, self.dtype)
        self.siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_name)
        self.t_config = self.transformer.config

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings
    @torch.no_grad()
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.t_config.patch_size
        vae_scale_factor_spatial = 8
        grid_height = height // (vae_scale_factor_spatial * p)
        grid_width = width // (vae_scale_factor_spatial * p)
        base_size_width = self.t_config.sample_width // p
        base_size_height = self.t_config.sample_height // p

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.t_config.attention_head_dim,
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
    def prepare_latent(
        self, 
        ref_videos, 
        driving_videos, 
        pixel_masks, 
        latent_masks, 
        timesteps,
        noisy_latent=None,
    ):
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

        ref_videos = torch.stack(ref_videos, dim=0).to(self.device, self.dtype) # (B, C, T, H, W)
        driving_videos = torch.stack(driving_videos, dim=0).to(self.device, self.dtype)  # (B, C, T, H, W)
        pixel_masks = torch.stack(pixel_masks, dim=0).to(self.device, self.dtype)  # (B, 1, T, H, W)
        latent_masks = torch.stack(latent_masks, dim=0).to(self.device, self.dtype)  # (B, 64, T', H', W')

        if noisy_latent is None:
            clean_latent = self.vae.encode(ref_videos).latent_dist.sample()  # (B, C, T', H', W_)
            noise = torch.randn_like(clean_latent, device=ref_videos.device, dtype=self.dtype)
            noisy_latent = self.scheduler.add_noise(
                clean_latent, 
                noise, 
                torch.tensor(timesteps, dtype=torch.int64, device=self.device)
            )
        else:
            noise = None

        # note: this is a no-op anyway with this scheduler, but would require looping for this impl
        # so we skip it
        # noisy_latent = self.scheduler.scale_model_input(noisy_latent, timesteps)

        # Cut out all pixels in the mask from the reference.
        # We leave the first frame intact.
        pixel_mask = pixel_masks[:, :, 1:, :, :]
        ref_videos[:, :, 1:, :, :] *= (1.0 - pixel_mask)

        # Apply darkening to the reference frames (after first)
        ref_videos[:, :, 1:, :, :] *= self.ref_frames_strength
        ref_latent = self.vae.encode(ref_videos).latent_dist.mode() * self.vae_scaling_factor_image

        lmk_latent = self.lmk_encoder.encode(driving_videos).latent_dist.mode()
        lmk_latent *= self.lmk_scaling_factor_image

        # concatenate along channel dimension (B, C, T', H', W')
        if self.explicit_mask_channels:
            model_input = torch.cat([noisy_latent, lmk_latent, ref_latent, latent_masks], dim=1)
        else:
            model_input = torch.cat([noisy_latent, lmk_latent, ref_latent], dim=1)

        return model_input, noise

    @torch.no_grad()
    def embed_reference_prompt(self, identity_images):
        """
        Embeds each identity image using the Siglip model.

        Args:
            identity_images: List of reference video images (C, H, W) with values in [0, 255]

        Returns:
            A tensor of shape [B, 729, 1152] where B is the number of reference videos.
        """
        
        imgs = self.siglip_normalize.preprocess(images=identity_images, do_resize=True, return_tensors="pt", do_convert_rgb=True)
        imgs = imgs.to(self.device, self.dtype)
        image_embeddings = self.siglip(**imgs).last_hidden_state  # torch.Size([B, 729, 1152])

        return image_embeddings.to(self.device, self.dtype)

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
            num_frames=model_inputs.shape[2],  # T'
            device=self.device,
        )

        model_inputs = model_inputs.permute(0, 2, 1, 3, 4) # Swap channels/frames
        noise_pred = self.transformer(
            hidden_states=model_inputs.to(self.transformer.device, self.dtype),
            encoder_hidden_states=image_embeddings.to(self.transformer.device, self.dtype),
            timestep=torch.tensor(timesteps, dtype=self.dtype, device=self.device),
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)  # Swap channels/frames back

        noise_pred = noise_pred.float()
        noise_gt = noise_gt.float()
        latent_masks = self.flatten_latent_mask(latent_masks).to(self.device)
        inpaint_loss = self.compute_inpaint_loss(noise_pred, noise_gt, latent_masks)
        optical_flow_loss = self.compute_masked_optical_flow_loss(
            noise_pred,
            noise_gt,
            latent_masks,
            optical_flow_masks,
        )
        
        final_loss = self.config.get("inpaint_lambda", 1.0) * inpaint_loss + self.config.get("optical_flow_lambda", 1.0) * optical_flow_loss
        return final_loss, inpaint_loss.item(), optical_flow_loss.item()
    
    @torch.no_grad()
    def full_inference(
        self,
        transformer,
        ref_videos, 
        driving_videos, 
        pixel_masks,
        identity_images,
        height,
        width,
    ):
        """
        Performs a full inference trajectory of the pipeline as it stands.
        This uses CFG with a guidance scale of 3.0 over 20 inference steps.
        Args:
            transformer: The unwrapped transformer model.
            ref_videos: List of reference video tensors (C, T, H, W)
            driving_videos: List of driving video tensors (C, T, H, W)
            pixel_masks: List of pixel mask tensors (1, T, H, W)
            identity_images: List of identity images (H, W)
            timesteps: List of timesteps
            height: Height of the input videos
            width: Width of the input videos
        Returns:
            A tensor of shape (B, C, T, H, W) where B is the batch size of ref_videos.

        """

        batch_size = len(ref_videos)
        guidance_scale = 3.0
        noisy_latent = torch.randn((batch_size, 16, 13, 60, 90), device=self.device)
        latent_masks = self.prepare_masks(pixel_masks)
        model_inputs, _ = self.prepare_latent(
            ref_videos, 
            driving_videos, 
            pixel_masks, 
            latent_masks, 
            [0] * batch_size, #not used
            noisy_latent=noisy_latent,
        )
        model_inputs = torch.cat([model_inputs, model_inputs], dim=0)  # Duplicate for CFG
        image_embeddings = self.embed_reference_prompt(identity_images)
        # extend to B*2 with zeros, for CFG
        image_embeddings = torch.cat([torch.zeros_like(image_embeddings), image_embeddings], dim=0)
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height=height,
            width=width,
            num_frames=model_inputs.shape[2],  # T'
            device=self.device,
        )

        timesteps, num_inference_steps = retrieve_timesteps_for_inference(
            self.inference_scheduler,
            self.inference_timesteps,
            self.device, 
            None,
        )

        # Swap channels/frames -> (B, T, C, H', W')
        model_inputs = model_inputs.permute(0, 2, 1, 3, 4)

        for i, t in enumerate(tqdm(timesteps)):
            model_inputs = model_inputs.to(transformer.device, self.dtype)
            image_embeddings = image_embeddings.to(transformer.device, self.dtype)

            timestep = t.expand(model_inputs.shape[0])
            noise_pred = transformer(
                hidden_states=model_inputs,
                encoder_hidden_states=image_embeddings,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # update the noisy latents, leave the rest of the model inputs untouched.
            latents = model_inputs[:batch_size, :, 0:16, :, :].float()
            latents = self.inference_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = torch.cat([latents, latents], dim=0).to(model_inputs.dtype)
            model_inputs[:, :, 0:16, :, :] = latents

        latents = latents[:batch_size].permute(0, 2, 1, 3, 4)  # (B, C, T', H', W')
        latents = latents * (1 / self.vae_scaling_factor_image)
        return self.vae.decode(latents).sample

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
        return latent_mask.mean(dim=1, keepdim=True)
    
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
