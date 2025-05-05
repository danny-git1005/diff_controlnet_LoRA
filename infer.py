import os
import argparse
import torch
import wandb
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

from config import Config
from utils.image_processors import get_processor_for_condition

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained ControlNet model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model")
    parser.add_argument("--condition_image", type=str, required=True, help="Path to the condition image")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for generation")
    parser.add_argument("--condition_type", type=str, choices=["canny", "depth", "pose", "seg"], default="seg",
                       help="Type of condition to use")
    parser.add_argument("--output_file", type=str, default="generated_image.png", help="Output file name")
    parser.add_argument("--log_to_wandb", action="store_true", help="Log results to wandb")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    
    # Update config from args
    if args.condition_type:
        config.condition_type = args.condition_type
    
    model_path = args.model_path or f"{config.output_dir}/final-model"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    
    # Initialize wandb if needed
    if args.log_to_wandb and config.log_wandb:
        if config.wandb_api_key:
            os.environ["WANDB_API_KEY"] = config.wandb_api_key
        wandb.init(project=config.project_name, job_type="inference")
    
    # Load the condition image
    condition_image = load_image(args.condition_image)
    
    # Process condition image
    processor = get_processor_for_condition(config.condition_type)
    processed_condition = processor(condition_image)
    
    # Load model
    try:
        # Try to load the complete pipeline first
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
    except Exception:
        # If that fails, load the base model and controlnet separately
        controlnet = ControlNetModel.from_pretrained(
            f"{model_path}/controlnet",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            config.base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    
    # Generate image
    print(f"Generating image with prompt: {args.prompt}")
    output_image = pipeline(
        args.prompt,
        processed_condition,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale
    ).images[0]
    
    # Save output image
    output_image.save(args.output_file)
    print(f"Image saved to {args.output_file}")
    
    # Log to wandb if enabled
    if args.log_to_wandb and config.log_wandb:
        wandb.log({
            "condition_image": wandb.Image(condition_image, caption="Condition Image"),
            "processed_condition": wandb.Image(processed_condition, caption="Processed Condition"),
            "output_image": wandb.Image(output_image, caption=f"Output: {args.prompt}"),
            "prompt": args.prompt
        })
        wandb.finish()

if __name__ == "__main__":
    main()
