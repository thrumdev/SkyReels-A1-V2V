from peft import PeftModel
import argparse

from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel

def main():
    parser = argparse.ArgumentParser(description="Merge and save LoRA weights")
    parser.add_argument("base_dir", type=str, help="Directory containing the base model")
    parser.add_argument("lora_dir", type=str, help="Path to the LoRA checkpoint")
    parser.add_argument("output_dir", type=str, help="Directory to save the merged model")
    args = parser.parse_args()

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.base_dir,
        subfolder="transformer",
    )

    transformer.patch_embed.expand_proj_channels(112)

    # Add mask channels
    transformer.config.in_channels = 112

    transformer = PeftModel.from_pretrained(
        transformer,
        model_id=args.lora_dir,
        is_trainable=False,
    )

    merged_model = transformer.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
