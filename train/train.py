from omegaconf import OmegaConf
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
import torch
import os
import time
import cv2
import numpy as np
import tqdm

from .dataloader import get_dataloader
from .pipeline import SkyReelsA1V2VInpaintPipeline

import wandb

@torch.no_grad()
def expand_mask(mask, expand_x, expand_y):
    """
    Expand the masked region in the mask tensor to include an amount of surrounding pixels in x and y directions.
    Args:
        mask (torch.Tensor): The mask tensor of shape (1, T, H, W)
        expand_x (int): The number of pixels to expand the mask by in the x direction (width).
        expand_y (int): The number of pixels to expand the mask by in the y direction (height).
    """

    if expand_x <= 0 and expand_y <= 0:
        return mask
    
    # rearrange mask to (T, 1, H, W)
    mask = mask.permute(1, 0, 2, 3)

    # Create a kernel for dilation
    # (out_channels, in_channels/groups, kernel_height, kernel_width)
    kernel = torch.ones((1, 1, expand_y * 2 + 1, expand_x * 2 + 1), device=mask.device)

    # Use dilation to expand the mask
    expanded_mask = torch.nn.functional.conv2d(mask.float(), kernel, padding=(expand_y, expand_x))
    
    # Threshold the result to create a binary mask
    expanded_mask = (expanded_mask > 0).float()

    # rearrange back to (1, T, H, W)
    expanded_mask = expanded_mask.permute(1, 0, 2, 3)

    return expanded_mask

def trim_batch_items(ref_videos, driving_videos, masks, optical_flow_masks, frames):
    """
    Trims/prunes each item in the batch to a random slice of 'frames' frames.
    If there are fewer actual frames than 'frames', repeats the final frame to pad.
    Modifies the lists in-place.
    """
    for i in range(len(ref_videos)):
        actual_frames = ref_videos[i].shape[1]
        if actual_frames > frames:
            start_frame = torch.randint(0, actual_frames - frames + 1, (1,)).item()
            ref_videos[i] = ref_videos[i][:, start_frame:start_frame + frames, :, :]
            driving_videos[i] = driving_videos[i][:, start_frame:start_frame + frames, :, :]
            masks[i] = masks[i][:, start_frame:start_frame + frames, :, :]
            optical_flow_masks[i] = optical_flow_masks[i][:, start_frame:start_frame + frames - 2, :, :]
        elif actual_frames < frames:
            pad_len = frames - actual_frames
            # Pad ref_video
            last_ref = ref_videos[i][:, -1:, :, :].repeat(1, pad_len, 1, 1)
            ref_videos[i] = torch.cat([ref_videos[i], last_ref], dim=1)
            # Pad driving_video
            last_driving = driving_videos[i][:, -1:, :, :].repeat(1, pad_len, 1, 1)
            driving_videos[i] = torch.cat([driving_videos[i], last_driving], dim=1)
            # Pad mask
            last_mask = masks[i][:, -1:, :, :].repeat(1, pad_len, 1, 1)
            masks[i] = torch.cat([masks[i], last_mask], dim=1)
            # Pad optical_flow_mask (T-2)
            last_optical = optical_flow_masks[i][:, -1:, :, :].repeat(1, pad_len, 1, 1)
            optical_flow_masks[i] = torch.cat([optical_flow_masks[i], last_optical], dim=1)
    return ref_videos, driving_videos, masks, optical_flow_masks

def expand_masks_randomly(masks, max_expand):
    """
    Expands each mask in the batch by a random amount up to max_expand in x and y directions.
    Modifies the masks list in-place.
    """
    for i in range(len(masks)):
        mask_expand_x = torch.randint(0, max_expand + 1, (1,)).item()
        mask_expand_y = torch.randint(0, max_expand + 1, (1,)).item()
        masks[i] = expand_mask(masks[i], mask_expand_x, mask_expand_y)
    return masks

class Trainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator()
        # Pass distributed parameters to get_dataloader
        self.dataloader = get_dataloader(
            config.data_dir, 
            config, 
        )
        self.wandb_enabled = config.get("wandb_enabled", False)
        if self.wandb_enabled and self.accelerator.is_main_process:
            wandb.init(
                project=config.get("wandb_project", "skyreels-a1-v2v"),
                name=config.get("wandb_name", None),
            )

        device = self.accelerator.device
        self.pipeline = SkyReelsA1V2VInpaintPipeline(config, device)
        self.pipeline.transformer.train()
        self.pipeline.transformer.requires_grad_(False)

        # LoRA PEFT setup
        lora_rank = config.get("lora_rank")
        restore_checkpoint = config.get("restore_checkpoint")

        if lora_rank is not None:
            print(f"Training LoRA (rank={lora_rank})")
            use_rslora = config.get("use_rslora", False)

            target_modules = []
            for name, module in self.pipeline.transformer.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    target_modules.append(name)

            if restore_checkpoint is None:
                lora_config = LoraConfig(r=lora_rank, use_rslora=use_rslora, target_modules=target_modules) 
                self.pipeline.transformer = get_peft_model(self.pipeline.transformer, lora_config)
            else:
                print("Restoring saved LoRa")
                self.pipeline.transformer = PeftModel.from_pretrained(
                    self.pipeline.transformer,
                    is_trainable=True,
                )
        else:
            print("Training without LoRA")
            self.pipeline.transformer.requires_grad_(True)

        if config.get("gradient_checkpointing", False):
            print("Enabling gradient checkpointing")
            # this take a module parameter (not used) and then value as a kwarg
            self.pipeline.transformer._set_gradient_checkpointing("", value=True)

        for name, param in self.pipeline.transformer.named_parameters():
            if param.requires_grad and self.accelerator.is_main_process:
                print(f"Trainable parameter: {name} - {param.shape}")

        # Create optimizer out of all trained parameters.
        trained_params = filter(lambda p: p.requires_grad, self.pipeline.transformer.parameters())

        lr = config.get("learning_rate", 1e-4)
        optimizer = torch.optim.AdamW(trained_params, lr=lr)

        if restore_checkpoint is not None:
            print("restoring optimizer state")
            optimizer_path = os.path.join(restore_checkpoint, "optimizer.pt")
            optimizer_state = torch.load(optimizer_path)
            optimizer.load_state_dict(optimizer_state)

        if config.get("compile", False):
            self.pipeline.transformer.compile_blocks()

        # Prepare model, dataloader, and optimizer for distributed/accelerated training
        self.pipeline.transformer, self.dataloader, self.optimizer = self.accelerator.prepare(self.pipeline.transformer, self.dataloader, optimizer)
        self.pipeline.transformer.__dict__["_orig_mod"] = "" # workaround for partial compilation

        # Validation dataloader
        self.validation_steps = config.get("validation_steps", 1000)
        self.validation_data_dir = config.get("validation_data_dir", None)
        self.validation_dataloader = None
        if self.validation_data_dir:
            self.validation_dataloader = get_dataloader(
                self.validation_data_dir, 
                config, 
                mode="val", 
            )
            self.validation_dataloader = self.accelerator.prepare(self.validation_dataloader)

        # Gradient accumulation setup
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self._step_in_accum = 0

    def batch_to_device(self, batch):
        # Pipeline expects lists of tensors, not batched tensors
        ref_videos = list(batch["ref_video"])
        driving_videos = list(batch["driving_video"])
        masks = list(batch["mask"])
        optical_flow_masks = list(batch["optical_flow_mask"])
        identity_images = list(batch["cropped_aligned_identity"])

        def to_device(list, dtype=None):
            if dtype is None:
                dtype = self.pipeline.dtype
            return [x.to(self.accelerator.device, dtype) for x in list]

        ref_videos = to_device(ref_videos)
        driving_videos = to_device(driving_videos)
        masks = to_device(masks)
        optical_flow_masks = to_device(optical_flow_masks)
        identity_images = to_device(identity_images, dtype=torch.float32)

        return ref_videos, driving_videos, masks, optical_flow_masks, identity_images

    def train_one_step(self, batch):
        ref_videos, driving_videos, masks, optical_flow_masks, identity_images = self.batch_to_device(batch)
        frames = 49

        # The dataloader ensures that all the batch items have the same shape.
        height, width = ref_videos[0].shape[2], ref_videos[0].shape[3]

        # Select a random slice of frames
        ref_videos, driving_videos, masks, optical_flow_masks = trim_batch_items(ref_videos, driving_videos, masks, optical_flow_masks, frames)

        masks = expand_masks_randomly(masks, self.config.get("max_mask_expand", 0))

        num_train_timesteps = self.pipeline.scheduler.config.num_train_timesteps
        timesteps = logit_normal_uniform_blend(
            num_samples=len(ref_videos),
            num_train_timesteps=num_train_timesteps,
            blend=self.config.get("logit_normal_blend", 0.6)
        )

        loss = self.pipeline.step_forward_and_loss(
            ref_videos,
            driving_videos,
            masks,
            optical_flow_masks,
            identity_images,
            timesteps,
            height,
            width,
        )
        loss = loss / self.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.item()
    
    def init_pbar(self):
        if self.accelerator.is_main_process:
            self.pbar = tqdm.tqdm(
                total=self.gradient_accumulation_steps, 
                desc="substeps",
                leave=False,
            )

    def step_pbar(self):
        if self.accelerator.is_main_process:
            self.pbar.update(1)

    def train(self):
        step = self.config.get("start_step", 0)
        max_steps = self.config.get("num_steps", 5000)
        save_frequency = self.config.get("save_frequency", 1000)
        self.optimizer.zero_grad()

        self.init_pbar()
        batch_losses = []

        while True:
            for batch in self.dataloader:
                if step >= max_steps:
                    print(f"Reached maximum training steps: {max_steps}. Stopping training.")
                    return max_steps
                batch_losses.append(self.train_one_step(batch))
                self._step_in_accum += 1
                self.step_pbar()
                if self._step_in_accum % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    grad_norm = self.gradient_norm()

                    self.optimizer.zero_grad()

                    batch_losses = torch.tensor(batch_losses, device=self.accelerator.device)
                    batch_loss = self.accelerator.gather(batch_losses).mean().item()

                    # Validation
                    avg_val_loss = None
                    if self.validation_steps and self.validation_steps > 0 and step % self.validation_steps == 0:
                        avg_val_loss = self.validate(step)

                    if self.config.wandb_enabled and self.accelerator.is_main_process:
                        wandb.log({"step": step, "loss": batch_loss, "val_loss": avg_val_loss, "grad_norm": grad_norm})

                    step += 1
                    self.init_pbar()

                    if self.accelerator.is_main_process:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] Step {step}/{max_steps}, Last loss: {batch_loss:.4f}")
                    if step % save_frequency == 0 and self.accelerator.is_main_process:
                        self.save(step)

                    batch_losses = []

    def gradient_norm(self):
        """
        Computes the gradient norm of the model parameters.
        Returns:
            float: The gradient norm.
        """
        total_norm = 0.0
        model = self.accelerator.unwrap_model(self.pipeline.transformer)
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def save(self, step):
        save_dir = os.path.join(self.config.get("save_dir", "checkpoints"), f"model_step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.pipeline.transformer)

        if self.config.get("lora_rank") is not None:
            # If using LoRA, save only the adapter weights
            model.save_pretrained(save_dir, save_adapter=True)
        else:
            model.save_pretrained(save_dir)

        optimizer_save = os.path.join(save_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_save)

        # Remove older checkpoints, keep only last max_checkpoints
        max_checkpoints = self.config.get("max_checkpoints", 5)
        parent_dir = self.config.get("save_dir", "checkpoints")
        # List all checkpoint directories matching model_step_*
        ckpt_dirs = [d for d in os.listdir(parent_dir) if d.startswith("model_step_") and os.path.isdir(os.path.join(parent_dir, d))]
        ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("_step_")[1]))
        if len(ckpt_dirs) > max_checkpoints:
            for old_ckpt in ckpt_dirs[:-max_checkpoints]:
                old_ckpt_path = os.path.join(parent_dir, old_ckpt)
                try:
                    # Remove the entire directory and its contents
                    import shutil
                    shutil.rmtree(old_ckpt_path)
                    print(f"Removed old checkpoint directory: {old_ckpt_path}")
                except Exception as e:
                    print(f"Error removing {old_ckpt_path}: {e}")

    def validate(self, step):
        if not self.validation_dataloader:
            return None
        
        if self.accelerator.is_main_process:
            print(f"Running validation... step={step}")

        self.pipeline.transformer.eval()
        val_losses = []
        first_item = None
        with torch.no_grad():
            data = self.validation_dataloader
            if self.accelerator.is_main_process:
                data = tqdm.tqdm(data, desc="validation", total=len(self.validation_dataloader))
            for batch in data:
                ref_videos, driving_videos, masks, optical_flow_masks, identity_images = self.batch_to_device(batch)
                frames = 49
                height, width = ref_videos[0].shape[2], ref_videos[0].shape[3]
                # Shared trimming/pruning
                ref_videos, driving_videos, masks, optical_flow_masks = trim_batch_items(ref_videos, driving_videos, masks, optical_flow_masks, frames)
                masks = expand_masks_randomly(masks, self.config.get("max_mask_expand", 0))
                num_train_timesteps = self.pipeline.scheduler.config.num_train_timesteps
                timesteps = [torch.randint(0, num_train_timesteps, (1,), dtype=torch.long).item() for _ in range(len(ref_videos))]

                if first_item is None:
                    first_item = {
                        "ref_video": ref_videos[0],
                        "driving_video": driving_videos[0],
                        "mask": masks[0],
                        "identity_image": identity_images[0],
                    }
                loss = self.pipeline.step_forward_and_loss(
                    ref_videos,
                    driving_videos,
                    masks,
                    optical_flow_masks,
                    identity_images,
                    timesteps,
                    height,
                    width,
                )
                val_losses.append(loss.cpu().item())

        # Gather losses from all processes
        loss_tensor = torch.tensor(val_losses, device="cpu").to(self.accelerator.device)
        gathered = self.accelerator.gather(loss_tensor)
        gathered = gathered.cpu().numpy().tolist()
        avg_val_loss = sum(gathered) / len(gathered) if gathered else None

        self.save_full_inference_example(step, first_item)

        self.pipeline.transformer.train()

        return avg_val_loss
    
    @torch.no_grad()
    def save_full_inference_example(self, step, item):
        if not self.accelerator.is_main_process:
            return
        
        print(f"Saving full inference example for step {step}...")

        height, width = item["ref_video"].shape[2], item["ref_video"].shape[3]
        transformer = self.accelerator.unwrap_model(self.pipeline.transformer)
        output = self.pipeline.full_inference(
            transformer,
            [item["ref_video"]],
            [item["driving_video"]],
            [item["mask"]],
            [item["identity_image"]],
            height,
            width,
        ).squeeze(0).to(self.accelerator.device)

        mask_grayscale = item["mask"].repeat(3, 1, 1, 1)  # Convert mask to 3-channel grayscale

        # Combine ref_video, output video, mask, landmarks in a H*2, W*2 grid.
        top = torch.cat([item["ref_video"], output], dim=3)  # (C, T, H, W*2)
        bottom = torch.cat([mask_grayscale, item["driving_video"]], dim=3)  # (C, T, H, W*2)
        combined = torch.cat([top, bottom], dim=2)  # (C, T, H*2, W*2)

        # If there are more than 20 files in the directory, remove the one with the smallest step number
        val_save_dir = self.config.get("val_save_dir", "denoised")
        os.makedirs(val_save_dir, exist_ok=True)
        if len(os.listdir(val_save_dir)) > 20:
            files = os.listdir(val_save_dir)
            files = [f for f in files if f.startswith("step_")]
            files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            if files:
                oldest_file = files[0]
                os.remove(os.path.join(val_save_dir, oldest_file))
                print(f"Removed old validation example: {oldest_file}")

        # Save the combined tensor as a video
        video = (combined.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)  # (T, H*2, W*2, C)
        save_path = os.path.join(val_save_dir, f"step_{step}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 16.0, (width*2, height*2))
        for frame in video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
    
def logit_normal_uniform_blend(num_samples, num_train_timesteps, blend):
    """
    Sample `num_samples` timesteps from a blended distribution of logit-normal and uniform.
    Args:
        num_samples (int): Number of timesteps to sample.
        num_train_timesteps (int): Total number of timesteps.
        blend (float): Blending factor between logit-normal and uniform.
    Returns:
        List[int]: Sampled timesteps, clamped to the range [0, num_train_timesteps - 1].
    """
    mu = 0.0
    sigma = 1.0
    # Logit-normal component
    z = torch.randn(num_samples) * sigma + mu
    x_logit = torch.sigmoid(z)
    # Uniform component
    x_uniform = torch.rand(num_samples)
    x_choice = torch.rand(num_samples)

    # Blend
    x_blend = torch.where(
        x_choice < blend,
        x_logit,
        x_uniform
    )

    # Scale to timesteps and quantize
    t = (x_blend * (num_train_timesteps - 1)).to(torch.int64)
    t = torch.clamp(t, 0, num_train_timesteps - 1)
    return list(t)

def main():
    import sys
    if len(sys.argv) > 1:
        config = OmegaConf.load(sys.argv[1])
    else:
        config = OmegaConf.create({})  
    print("Loaded config:", config)
    trainer = Trainer(config)

    final_step = trainer.train()
    trainer.save(final_step)

if __name__ == "__main__":
    main()
