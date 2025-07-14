from omegaconf import OmegaConf
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import torch
import os
import time

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
        if lora_rank is not None:
            print(f"Training LoRA (rank={lora_rank})")
            use_rslora = config.get("use_rslora", False)

            target_modules = []
            for name, module in self.pipeline.transformer.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    target_modules.append(name)
            lora_config = LoraConfig(r=lora_rank, use_rslora=use_rslora, target_modules=target_modules) 
            self.pipeline.transformer = get_peft_model(self.pipeline.transformer, lora_config)
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
        optimizer = torch.optim.AdamW(trained_params, lr=config.get("learning_rate", 1e-4))

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

        # Gradient accumulation setup
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self._step_in_accum = 0

    def train_one_step(self, batch):

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

        frames = 49

        # The dataloader ensures that all the batch items have the same shape.
        height, width = ref_videos[0].shape[2], ref_videos[0].shape[3]

        # Select a random slice of frames
        ref_videos, driving_videos, masks, optical_flow_masks = trim_batch_items(ref_videos, driving_videos, masks, optical_flow_masks, frames)

        masks = expand_masks_randomly(masks, self.config.get("max_mask_expand", 0))

        num_train_timesteps = self.pipeline.scheduler.config.num_train_timesteps
        timesteps = [torch.randint(0, num_train_timesteps, (1,), dtype=torch.long).item() for _ in range(len(ref_videos))]

        loss = self.pipeline.step_forward_and_loss(
            ref_videos,
            driving_videos,
            masks,
            optical_flow_masks,
            identity_images,
            timesteps,
            height,
            width,
            frames
        )
        loss = loss / self.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.item()

    def train(self):
        step = 0
        max_steps = self.config.get("num_steps", 5000)
        save_frequency = self.config.get("save_frequency", 1000)
        self.optimizer.zero_grad()
        while True:
            for batch in self.dataloader:
                if step >= max_steps:
                    print(f"Reached maximum training steps: {max_steps}. Stopping training.")
                    return max_steps
                loss = self.train_one_step(batch)
                self._step_in_accum += 1
                if self._step_in_accum % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # Only log to wandb on the main process
                    if self.config.wandb_enabled and self.accelerator.is_main_process:
                        wandb.log({"step": step, "loss": loss})
                    # Validation
                    if self.validation_steps and self.validation_steps > 0 and step % self.validation_steps == 0 and step > 0:
                        avg_val_loss = self.validate()
                        if avg_val_loss is not None and self.config.wandb_enabled and self.accelerator.is_main_process:
                            wandb.log({"step": step, "val_loss": avg_val_loss})
                    step += 1
                    if step % 10 == 0 and self.accelerator.is_main_process:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] Step {step}/{max_steps}, Last loss: {loss:.4f}")
                    if step % save_frequency == 0 and self.accelerator.is_main_process:
                        self.save(step)

    def save(self, step):
        save_dir = os.path.join(self.config.get("save_dir", "checkpoints"), f"model_step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.pipeline.transformer)

        if self.config.get("lora_rank") is not None:
            # If using LoRA, save only the adapter weights
            model.save_pretrained(save_dir, save_adapter=True)
        else:
            model.save_pretrained(save_dir)

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

    def validate(self):
        if not self.validation_dataloader:
            return None
        self.pipeline.transformer.eval()
        val_losses = []
        with torch.no_grad():
            for batch in self.validation_dataloader:
                ref_videos = list(batch["ref_video"])
                driving_videos = list(batch["driving_video"])
                masks = list(batch["mask"])
                optical_flow_masks = list(batch["optical_flow_mask"])
                identity_images = list(batch["cropped_aligned_identity"])
                frames = 49
                height, width = ref_videos[0].shape[2], ref_videos[0].shape[3]
                # Shared trimming/pruning
                ref_videos, driving_videos, masks, optical_flow_masks = trim_batch_items(ref_videos, driving_videos, masks, optical_flow_masks, frames)
                masks = expand_masks_randomly(masks, self.config.get("max_mask_expand", 0))
                num_train_timesteps = self.pipeline.scheduler.config.num_train_timesteps
                timesteps = [torch.randint(0, num_train_timesteps, (1,), dtype=torch.long).item() for _ in range(len(ref_videos))]
                loss = self.pipeline.step_forward_and_loss(
                    ref_videos,
                    driving_videos,
                    masks,
                    optical_flow_masks,
                    identity_images,
                    timesteps,
                    height,
                    width,
                    frames
                )
                val_losses.append(loss.item())
        # Gather losses from all processes
        gathered = self.accelerator.gather(torch.tensor(val_losses, device=self.accelerator.device))
        gathered = gathered.cpu().numpy().tolist()
        self.pipeline.transformer.train()
        avg_val_loss = sum(gathered) / len(gathered) if gathered else None
        return avg_val_loss

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
