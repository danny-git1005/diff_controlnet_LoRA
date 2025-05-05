import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ControlNetDataset(Dataset):
    def __init__(self, image_dir, condition_dir, prompts=None, processor=None, 
                 resolution=512, use_text_condition=True):
        self.image_dir = image_dir
        self.condition_dir = condition_dir
        self.prompts = prompts or {}
        self.processor = processor
        self.resolution = resolution
        self.use_text_condition = use_text_condition
        
        # Get image file list
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Initialize preprocessing transforms
        self.transforms = A.Compose([
            A.Resize(height=resolution, width=resolution),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Generate or load condition image
        # If the condition already exists, load it. Otherwise, generate it.
        condition_file = os.path.splitext(image_file)[0] + ".png"
        condition_path = os.path.join(self.condition_dir, condition_file)

        if os.path.exists(condition_path):
            condition_image = Image.open(condition_path).convert("RGB")
        else:
            # If condition doesn't exist and we have a processor, create it
            if self.processor:
                condition_image = self.processor(image)
                # Save the condition for future use
                os.makedirs(self.condition_dir, exist_ok=True)
                condition_image.save(condition_path)
            else:
                # If no processor, just use the original image as the condition
                condition_image = image.copy()

        # Get prompt if using text conditions
        prompt_text = "" # 初始化一個用於回傳的 prompt 字串
        if self.use_text_condition:
            prompt_data = self.prompts.get(image_file, "") # 從 prompts 字典中獲取原始數據，可能是字串或字典
            
            # --- 最小修改在這裡 ---
            if isinstance(prompt_data, dict) and "description" in prompt_data:
                # 如果獲取到的是一個字典，並且包含 "description" 鍵
                prompt_text = prompt_data["description"]
            elif isinstance(prompt_data, str):
                # 如果獲取到的是字串 (例如預設的 "")
                prompt_text = prompt_data
            else:
                # 處理其他意外的數據類型，fallback 到空字串
                print(f"Warning: Unexpected prompt data type or format for {image_file}: {type(prompt_data)}. Using empty string.")
                prompt_text = ""
            # --- 修改結束 ---

        # Apply transforms
        image_np = np.array(image)
        condition_np = np.array(condition_image)

        transformed_image = self.transforms(image=image_np)["image"]
        transformed_condition = self.transforms(image=condition_np)["image"]

        return {
            "images": transformed_image,
            "condition_images": transformed_condition,
            "prompts": prompt_text, # 現在這裡使用一個確保是字串的變數
            "image_path": image_path,
            "condition_path": condition_path
        }
