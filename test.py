import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

from config import Config
from utils.image_processors import get_processor_for_condition

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference and visualize results")
    parser.add_argument("--input_dir", type=str, default="E:/Danny/phase2_diffusion_mdoel/data/japna_train_data/infer_example/tw_mask_img",help="Directory containing input images")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory containing ground truth images (optional)")
    parser.add_argument("--model_path", type=str, default="E:/Danny/phase2_diffusion_mdoel/diff_controlnet/outputs_1/final-model")
    parser.add_argument("--output_dir", type=str, default="E:/Danny/phase2_diffusion_mdoel/data/japna_train_data/infer_example/pred", help="Output directory for visualizations")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for generation")
    parser.add_argument("--condition_type", type=str, choices=["canny", "depth", "pose", "seg"], default=None,
                       help="Type of condition to use")
    parser.add_argument("--image_size", type=int, default=512, help="Size to resize images (will be applied to both width and height)")
    return parser.parse_args()

def load_model(config, model_path):
    """Load the ControlNet model from the specified path with improved error handling"""
    try:
        print(f"Attempting to load model from {model_path}")
        
        # Check if the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
            
        # Try to load the complete pipeline first
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            local_files_only=True  # Try to use local files only first
        )
        print("Successfully loaded full pipeline from local files")
    except Exception as e:
        print(f"Could not load full pipeline from local files: {e}")
        
        try:
            # Try loading again but allow downloading
            print("Trying to load from HuggingFace Hub...")
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                local_files_only=False
            )
            print("Successfully loaded full pipeline from HuggingFace Hub")
        except Exception as e:
            print(f"Could not load full pipeline from HuggingFace either: {e}")
            print("Attempting to load base model and controlnet separately...")
            
            controlnet_path = os.path.join(model_path, "controlnet")
            
            # Check if controlnet path exists and has the required files
            if not os.path.exists(controlnet_path) or not os.path.exists(os.path.join(controlnet_path, "config.json")):
                # Check if there are any subdirectories that might contain the controlnet
                subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
                print(f"Available directories in {model_path}: {subdirs}")
                
                # Try to find a directory that might be the controlnet
                for subdir in subdirs:
                    potential_path = os.path.join(model_path, subdir)
                    if os.path.exists(os.path.join(potential_path, "config.json")):
                        print(f"Found potential controlnet at {potential_path}")
                        controlnet_path = potential_path
                        break
            
            try:
                print(f"Loading controlnet from {controlnet_path}")
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    local_files_only=True  # Try local first
                )
            except Exception as controlnet_error:
                print(f"Could not load controlnet locally: {controlnet_error}")
                print("Trying to load controlnet from HuggingFace Hub...")
                
                try:
                    # Attempt to load from HuggingFace
                    controlnet = ControlNetModel.from_pretrained(
                        controlnet_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        local_files_only=False
                    )
                except Exception as online_controlnet_error:
                    print(f"Failed to load controlnet: {online_controlnet_error}")
                    raise RuntimeError("Could not load the ControlNet model locally or from HuggingFace Hub. "
                                      "Please check your model path and internet connection.")
            
            try:
                print(f"Loading base model from {config.base_model}")
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    config.base_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    local_files_only=True  # Try local first
                )
            except Exception as base_model_error:
                print(f"Could not load base model locally: {base_model_error}")
                print("Trying to load base model from HuggingFace Hub...")
                
                try:
                    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                        config.base_model,
                        controlnet=controlnet,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        safety_checker=None,
                        local_files_only=False
                    )
                except Exception as online_base_model_error:
                    print(f"Failed to load base model: {online_base_model_error}")
                    raise RuntimeError("Could not load the base model locally or from HuggingFace Hub. "
                                      "Please check your model path and internet connection.")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    
    return pipeline

def resize_image(image, target_size):
    """Resize an image to the target size while preserving aspect ratio"""
    width, height = image.size
    
    # Determine the scaling factor to maintain aspect ratio
    aspect_ratio = width / height
    
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new blank image with the target size
    new_image = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    
    # Paste the resized image onto the blank image, centered
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image

def visualize_results(condition_image, ground_truth, prediction, output_path, input_filename):
    """Create a visualization comparing condition, ground truth, and prediction"""
    if ground_truth is not None:
        # Create a three-panel visualization with ground truth
        plt.figure(figsize=(15, 5))
        
        # Display condition image
        plt.subplot(1, 3, 1)
        plt.imshow(condition_image)
        plt.title("Condition Image")
        plt.axis("off")
        
        # Display ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth)
        plt.title("Ground Truth")
        plt.axis("off")
        
        # Display prediction
        plt.subplot(1, 3, 3)
        plt.imshow(prediction)
        plt.title("Model Prediction")
        plt.axis("off")
    else:
        # Create a two-panel visualization without ground truth
        plt.figure(figsize=(10, 5))
        
        # Display condition image
        plt.subplot(1, 2, 1)
        plt.imshow(condition_image)
        plt.title("Condition Image")
        plt.axis("off")
        
        # Display prediction
        plt.subplot(1, 2, 2)
        plt.imshow(prediction)
        plt.title("Model Prediction")
        plt.axis("off")
    
    plt.suptitle(f"Comparison for {input_filename}")
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def process_batch(input_dir, gt_dir, output_dir, model_path, prompt, condition_type, image_size):
    """Process a batch of images and visualize the results"""
    config = Config()
    
    # Update config from args
    if condition_type:
        config.condition_type = condition_type
    
    model_path = model_path or f"{config.output_dir}/final-model"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    pipeline = load_model(config, model_path)
    
    # Get processor for condition
    processor = get_processor_for_condition(config.condition_type)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Images will be resized to {image_size}x{image_size}")
    
    # Check if ground truth directory is provided
    has_ground_truth = gt_dir is not None and os.path.exists(gt_dir)
    if not has_ground_truth:
        print("No ground truth directory provided. Proceeding without ground truth comparison.")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Construct paths
        input_path = os.path.join(input_dir, img_file)
        
        # Load and process condition image
        original_condition_image = load_image(input_path)
        condition_image = resize_image(original_condition_image, image_size)
        
        # Handle ground truth if available
        ground_truth = None
        if has_ground_truth:
            gt_path = os.path.join(gt_dir, img_file)  # Assuming matching filenames
            if os.path.exists(gt_path):
                original_ground_truth = load_image(gt_path)
                ground_truth = resize_image(original_ground_truth, image_size)
            else:
                print(f"Warning: No ground truth found for {img_file}")
        
        # Process condition image
        processed_condition = processor(condition_image)
        
        # Generate image
        output_image = pipeline(
            prompt,
            processed_condition,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale
        ).images[0]
        
        # Visualize results
        vis_output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_comparison.png")
        visualize_results(condition_image, ground_truth, output_image, vis_output_path, img_file)
        
        # Also save the prediction separately
        prediction_output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_prediction.png")
        output_image.save(prediction_output_path)

def main():
    args = parse_args()
    
    print(f"Processing images from {args.input_dir}")
    if args.gt_dir:
        print(f"Using ground truth from {args.gt_dir}")
    print(f"Saving visualizations to {args.output_dir}")
    
    process_batch(
        args.input_dir,
        args.gt_dir,
        args.output_dir,
        args.model_path,
        args.prompt,
        args.condition_type,
        args.image_size
    )
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()