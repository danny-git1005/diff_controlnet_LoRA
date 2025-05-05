# -*- coding: utf-8 -*-
import os
import json
import copy
import argparse
import torch
import wandb
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Diffusers 和相關函式庫
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs # 為了 find_unused_parameters
from torch.utils.data import DataLoader
from peft import LoraConfig # <-- Import LoraConfig
from peft import get_peft_model
from peft import PeftModel

# 自訂模組 (需要你有這些檔案)
from config import Config # 假設你的設定檔名為 config.py
from utils.dataset import ControlNetDataset # 假設你的資料集類別在 utils/dataset.py

# 環境變數設定
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # 避免 TensorFlow OneDNN 相關警告


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="Train a ControlNet model using LoRA")
    parser.add_argument("--config", type=str, default="config.py", help="Path to config file")
    # --- 允許命令行覆蓋 Config 中的部分設定 ---
    parser.add_argument("--condition_type", type=str, choices=["canny", "depth", "pose", "seg"], default=None,
                        help="Override config: Type of condition to use (aligns with config)")
    parser.add_argument("--use_text_condition", default=None, type=lambda x: (str(x).lower() == 'true'),
                        help="Override config: Use text prompts (True/False)")
    # --- LoRA 參數覆蓋 ---
    parser.add_argument("--use_lora", default=None, type=lambda x: (str(x).lower() == 'true'),
                        help="Override config: Use LoRA training (True/False)")
    parser.add_argument("--lora_rank", type=int, default=None, help="Override config: LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Override config: LoRA alpha")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a saved training state checkpoint (directory). Use this to resume training.")
    parser.add_argument("--controlnet_checkpoint_path", type=str, default=None,help="Path to a specific ControlNet model checkpoint (directory). Use this to load only CN weights.")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- 載入設定 ---
    # 這部分假設 config.py 裡有一個 Config 類別
    # 你可能需要根據你的 config.py 結構調整這部分
    try:
        # 假設 Config() 直接可以使用
        config = Config()
        print("Loaded configuration from default config.py")
    except ImportError:
        print(f"Error: Could not import Config from config.py.")
        print("Please ensure config.py exists and defines a Config class.")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # --- 使用命令行參數更新設定 (如果提供了) ---
    if args.condition_type is not None:
        config.condition_type = args.condition_type
        print(f"Overridden config: condition_type = {config.condition_type}")
    if args.use_text_condition is not None:
        config.use_text_condition = args.use_text_condition
        print(f"Overridden config: use_text_condition = {config.use_text_condition}")
    if args.use_lora is not None:
        config.use_lora = args.use_lora
        print(f"Overridden config: use_lora = {config.use_lora}")
    if args.lora_rank is not None:
        config.lora_rank = args.lora_rank
        # 通常 lora_alpha 會跟著 rank 變，除非特別指定
        if args.lora_alpha is None:
            config.lora_alpha = args.lora_rank
        print(f"Overridden config: lora_rank = {config.lora_rank}")
    if args.lora_alpha is not None:
        config.lora_alpha = args.lora_alpha
        print(f"Overridden config: lora_alpha = {config.lora_alpha}")

    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
        print(f"Overridden config: resume_from_checkpoint = {config.resume_from_checkpoint}")
    if args.controlnet_checkpoint_path is not None:
        config.controlnet_checkpoint_path = args.controlnet_checkpoint_path
        print(f"Overridden config: controlnet_checkpoint_path = {config.controlnet_checkpoint_path}")


    # --- 檢查是否啟用 LoRA ---
    if not hasattr(config, 'use_lora') or not config.use_lora:
        print("\nError: LoRA training is not enabled.")
        print("Please set 'use_lora = True' in your config file or pass '--use_lora True' via command line.")
        return

    print(f"\n--- Starting ControlNet LoRA Training ---")
    print(f"  Base Model: {config.base_model}")
    print(f"  ControlNet Model: {config.controlnet_model}")
    if args.resume_from_checkpoint:
        print(f" Resume from Checkpoint: {args.resume_from_checkpoint}")
    if args.controlnet_checkpoint_path:
        print(f" Load ControlNet Checkpoint: {args.controlnet_checkpoint_path}")

    print(f"  Condition Type: {config.condition_type if hasattr(config, 'condition_type') else 'Not Specified'}")
    print(f"  Use Text Condition: {config.use_text_condition}")
    print(f"  LoRA Rank: {config.lora_rank}")
    print(f"  LoRA Alpha: {config.lora_alpha}")
    print(f"  Mixed Precision: {config.mixed_precision}")
    print(f"  Output Directory: {config.output_dir}")
    print("-" * 40)

    # --- 建立輸出目錄 ---
    os.makedirs(config.output_dir, exist_ok=True)

    # --- 設定 Accelerator ---
    # If using DDP and getting errors about unused parameters, find_unused_parameters can help
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.log_wandb else None, # 整合 wandb logging
        kwargs_handlers=[ddp_kwargs] # Handle potential DDP issues
    )
    if accelerator.is_main_process and config.log_wandb:
        try:
            if config.wandb_api_key:
                os.environ["WANDB_API_KEY"] = config.wandb_api_key
            
            wandb_entity = getattr(config, 'wandb_entity', 'default_entity')
            os.environ["WANDB_ENTITY"] = wandb_entity  # 給 wandb tracker 使用

            accelerator.init_trackers(
                project_name=config.project_name,
                config=vars(config)  # 記錄所有參數到 W&B config
            )
            print("Weights & Biases initialized via Accelerator.")
        except Exception as e:
            print(f"Could not initialize WandB via Accelerator: {e}")
            config.log_wandb = False
    print(f"Accelerator device: {accelerator.device}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Mixed precision: {accelerator.mixed_precision}")

    # --- 設定資料類型 ---
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # --- 載入 ControlNet 基礎模型 ---
    try:
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_model,
            torch_dtype=weight_dtype # 使用基於 accelerator 的 dtype
        )
        print(f"Loaded ControlNet base model from {config.controlnet_model}")
    except Exception as e:
        print(f"Error loading ControlNet model: {e}")
        return

    # --- 載入 Stable Diffusion Pipeline ---
    try:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            config.base_model,
            controlnet=controlnet, # 傳入基礎 ControlNet 結構
            torch_dtype=weight_dtype, # 使用基於 accelerator 的 dtype
            low_cpu_mem_usage=False # 通常在訓練時設為 False 避免問題
        )
        # 關閉 Safety Checker (常見於訓練)
        pipeline.safety_checker = None
        # pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images)) # 另一種關閉方式
        print(f"Loaded Stable Diffusion pipeline from {config.base_model}")
    except Exception as e:
        print(f"Error loading Stable Diffusion pipeline: {e}")
        return

    # --- 移動非訓練組件到目標設備 ---
    # VAE, Text Encoder, UNet 在 LoRA 訓練中不被訓練，提前移動
    pipeline.vae.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipeline.unet.to(accelerator.device, dtype=weight_dtype)
    print("Moved VAE, Text Encoder, UNet to accelerator device.")

    # --- 凍結非訓練模型的權重 ---
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # 凍結 ControlNet 的原始權重 (在加入 LoRA adapter 之前)
    controlnet.requires_grad_(False)
    print("Frozen weights for VAE, Text Encoder, UNet, and base ControlNet.")

    # --- 為 ControlNet 添加 LoRA Adapters ---
    # 設定 LoRA 層
    lora_controlnet_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian", # 或其他初始化方法，例如 "pissa"
        target_modules=[
            # Transformer 注意力層的標準目標
            "to_q",
            "to_k",
            "to_v",
            "to_out.0", # 正確地指定 ModuleList 中的線性層
            # Transformer 前饋網路層的標準目標
            "ff.net.0.proj", # GEGLU 內的線性層
            "ff.net.2",      # 最終線性層
            # 可選：ResNet 區塊中的時間嵌入投影層
            # "time_emb_proj", # 取消註解即可包含此層
        ],
        lora_dropout=getattr(config, 'lora_dropout', 0.1), # 從設定讀取 dropout，若無則預設 0.1
        # bias="none" # 或 "all" 或 "lora_only"。考慮使用 'none' 或 'lora_only' 以提高效率
    )

    try:
        # --- MODIFICATION: Wrap the controlnet using get_peft_model ---
        # This replaces the controlnet object with a PeftModel wrapper
        controlnet = get_peft_model(controlnet, lora_controlnet_config)
        print(f"Successfully wrapped ControlNet with PEFT LoRA adapters.")

        # --- MODIFICATION: Optional - Print trainable parameters using PEFT's method ---
        if accelerator.is_main_process:
            print("\nTrainable Parameters (PEFT):")
            controlnet.print_trainable_parameters() # PEFT helper function
        # --- MODIFICATION: Removed the controlnet.add_adapter() call as get_peft_model handles it ---

    except ValueError as ve: # Catch specific error related to target_modules
        print(f"Error wrapping ControlNet with PEFT: {ve}")
        print("This often means the 'target_modules' in LoraConfig are incorrect or not found.")
        print("Please carefully check the module names within your ControlNet structure.")
        # 提示用戶檢查模型結構
        if accelerator.is_main_process:
            print("\nInspecting ControlNet Model Structure (use this to find target_modules):")
            # Print all module names and types recursively for detailed inspection
            for name, module in accelerator.unwrap_model(controlnet).named_modules(): # Use unwrap_model if needed
                print(f"- {name}: {type(module)}")
        return
    except Exception as e:
        print(f"Error wrapping ControlNet with PEFT: {e}")
        traceback.print_exc() # Print full traceback for other errors
        return


    # --- [可選] 驗證可訓練參數 ---
    if accelerator.is_main_process:
        print("\nTrainable Parameters (should be LoRA layers):")
        trainable_param_count = 0
        all_param_count = 0
        for name, param in controlnet.named_parameters():
            all_param_count += param.numel()
            if param.requires_grad:
                print(f"  [Trainable] {name} ({param.numel()})")
                trainable_param_count += param.numel()
        print("-" * 20)
        print(f"Total Parameters (ControlNet): {all_param_count:,}")
        print(f"Trainable Parameters (LoRA):   {trainable_param_count:,}")
        if all_param_count > 0:
             print(f"Trainable Ratio: {trainable_param_count / all_param_count:.4%}")
        print("-" * 20)

    # --- 載入文字提示 (如果使用) ---
    prompts = {}
    if config.use_text_condition:
        prompt_file = getattr(config, 'prompt_file', 'prompts.json') # 從 config 讀取，預設 prompts.json
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r', encoding="utf-8") as f:
                    prompts = json.load(f)
                print(f"Loaded {len(prompts)} prompts from {prompt_file}")
            except Exception as e:
                print(f"Warning: Could not load prompts from {prompt_file}: {e}")
        else:
             print(f"Warning: Prompt file {prompt_file} not found. Text prompts might be missing.")

    # --- 建立資料集和資料載入器 ---
    try:
        train_dataset = ControlNetDataset(
            image_dir=config.image_dir,
            condition_dir=config.condition_dir,
            prompts=prompts, # 傳入載入的提示
            resolution=config.resolution,
            use_text_condition=config.use_text_condition
        )
        if len(train_dataset) == 0:
            print(f"Error: No data found in image_dir '{config.image_dir}' or condition_dir '{config.condition_dir}'. Please check paths.")
            return
        print(f"Created dataset with {len(train_dataset)} samples.")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # --- [可選] 在主進程顯示範例資料 ---
    if accelerator.is_local_main_process: # 僅在本地主進程顯示
        try:
            sample_idx = 0
            sample = train_dataset[sample_idx]
            image = sample["images"]
            condition_image = sample["condition_images"]
            prompt = sample["prompts"]

            # 將 Tensor 轉回 PIL Image 以便顯示
            if isinstance(image, torch.Tensor): image = TF.to_pil_image(image)
            if isinstance(condition_image, torch.Tensor): condition_image = TF.to_pil_image(condition_image)

            print(f"\nDisplaying Sample {sample_idx}:")
            print(f"  Prompt: '{prompt}'")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image); axes[0].set_title("Input Image"); axes[0].axis("off")
            axes[1].imshow(condition_image); axes[1].set_title("Condition Image"); axes[1].axis("off")
            plt.tight_layout();
            # 嘗試保存圖像而不是顯示，以避免 GUI 問題
            sample_fig_path = os.path.join(config.output_dir, "sample_data.png")
            plt.savefig(sample_fig_path)
            print(f"  Saved sample data visualization to {sample_fig_path}")
            plt.close(fig) # 關閉圖形，釋放資源
        except Exception as e:
            print(f"Warning: Could not display/save sample image: {e}")


    # --- 建立 DataLoader ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 4), # 從設定讀取，預設 4
        pin_memory=True # 通常能加速資料轉移
    )

    # --- 設定優化器 ---
    # 過濾參數，只優化可訓練的 LoRA 參數
    try:
        trainable_params = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
        if not trainable_params:
             print("Error: No trainable parameters found in ControlNet after adding LoRA adapter.")
             print("This might indicate an issue with LoraConfig target_modules or the adapter addition process.")
             return

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(getattr(config, 'adam_beta1', 0.9), getattr(config, 'adam_beta2', 0.999)),
            weight_decay=getattr(config, 'adam_weight_decay', 1e-2),
            eps=getattr(config, 'adam_epsilon', 1e-8),
        )
        print("Optimizer (AdamW) created for LoRA parameters.")
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        return

    # --- 設定學習率排程器 ---
    lr_scheduler_type = getattr(config, 'lr_scheduler', "cosine") # 從設定讀取，預設 cosine
    num_warmup_steps = getattr(config, 'lr_warmup_steps', 0) # 從設定讀取，預設 0

    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * accelerator.num_processes, # 考慮多 GPU warmup
        num_training_steps=config.max_train_steps * accelerator.num_processes, # 總訓練步數
    )
    print(f"Learning rate scheduler ({lr_scheduler_type}) created.")

    # --- 使用 Accelerator 準備模型、優化器、資料載入器和排程器 ---
    # 重要：只準備包含可訓練參數的模型 (controlnet)
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    print("Accelerator prepared ControlNet (LoRA), Optimizer, DataLoader, and LR Scheduler.")

    # --- 計算訓練總輪數 (Epochs) ---
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    num_train_epochs = (config.max_train_steps + num_update_steps_per_epoch - 1) // num_update_steps_per_epoch
    print(f"\n--- Training Details ---")
    print(f"  Total Training Steps: {config.max_train_steps}")
    print(f"  Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size per Device: {config.train_batch_size}")
    print(f"  Total Batch Size (across all devices): {config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps}")
    print(f"  Number of Epochs: {num_train_epochs}")
    print(f"  Updates per Epoch: {num_update_steps_per_epoch}")
    print("-" * 40)

    # --- 訓練迴圈 ---
    global_step = 0
    first_epoch = 0 # TODO: Add support for resuming training from checkpoint

    # 進度條只在主進程顯示
    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )

    best_loss = float("inf")

    print("\n--- Starting Training Loop ---")
    for epoch in range(first_epoch, num_train_epochs):
        controlnet.train() # 設定 ControlNet (含 LoRA) 為訓練模式
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # 使用 Accelerator 進行梯度累積
            with accelerator.accumulate(controlnet):
                # --- 準備輸入資料 ---
                # 將圖像轉為 float 並標準化到 [0, 1]
                images = batch["images"].to(dtype=torch.float32) / 255.0
                condition_images = batch["condition_images"].to(dtype=torch.float32) / 255.0

                # 將資料移動到目標設備並轉換類型 (根據混合精度設定)
                images = images.to(device=accelerator.device, dtype=weight_dtype)
                condition_images = condition_images.to(device=accelerator.device, dtype=weight_dtype)
                # --- 編碼圖像至 Latent Space (使用 VAE) ---
                # VAE 不需要梯度
                with torch.no_grad():
                    # 注意：pipeline.vae 已經在目標設備和 dtype 上
                    latents = pipeline.vae.encode(images).latent_dist.sample()
                    # 使用 VAE 的 scaling factor
                    latents = latents * pipeline.vae.config.scaling_factor

                # --- 產生雜訊和時間步 ---
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # 為 batch 中的每個樣本隨機選取時間步
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # 將雜訊添加到 Latent 中
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                # --- 獲取文字嵌入 (如果使用) ---
                if config.use_text_condition:
                    prompts = batch["prompts"]
                else:
                    prompts = [""] * bsz # 如果不用文字，則使用空提示

                # Text Encoder 不需要梯度
                with torch.no_grad():
                    # 注意：pipeline.tokenizer 和 pipeline.text_encoder 已在目標設備和 dtype 上
                    text_inputs = pipeline.tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(accelerator.device) # Tokenizer 輸出也需移到設備
                    # 獲取 text encoder 的最後隱藏層狀態
                    encoder_hidden_states = pipeline.text_encoder(text_inputs.input_ids)[0]
                    # 確保類型匹配
                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)


                # --- 前向傳播: ControlNet + UNet ---
                # ControlNet 需要梯度 (因為 LoRA 層是可訓練的)
                # 使用 accelerator 包裝後的 controlnet
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition_images,
                    return_dict=False, # 直接獲取 tuple 輸出
                )

                # 注意：pipeline.unet 已在目標設備和 dtype 上
                # 預測雜訊殘差
                model_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples # 確保 dtype 匹配
                    ] if isinstance(down_block_res_samples, (list, tuple)) else down_block_res_samples.to(dtype=weight_dtype),
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype) # 確保 dtype 匹配
                ).sample

                # --- 計算損失 ---
                # 使用 MSE 損失比較預測雜訊和實際添加的雜訊
                # 確保比較時使用 float32 以增加穩定性
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # 匯總所有進程的損失以供紀錄 (用於顯示平均損失)
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # --- 反向傳播 ---
                accelerator.backward(loss)

                # --- 更新權重 (只在梯度同步時執行) ---
                if accelerator.sync_gradients:
                    # 梯度裁剪 (可選但推薦)
                    max_grad_norm = getattr(config, 'max_grad_norm', 1.0) # 從設定讀取，預設 1.0
                    if max_grad_norm is not None:
                         # 只裁剪 controlnet 的可訓練參數 (LoRA)
                         accelerator.clip_grad_norm_(trainable_params, max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # --- 日誌記錄與進度更新 (只在梯度同步後執行) ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 準備日誌資訊
                logs = {"step_loss": avg_loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                # 使用 accelerator 記錄到 wandb (如果啟用)
                if config.log_wandb:
                    accelerator.log({"train_loss_step": avg_loss.item(), "lr": logs["lr"]}, step=global_step)

                # --- 保存檢查點 (LoRA 權重) ---
                save_steps = getattr(config, 'save_steps', 4) # 從設定讀取，預設 4
                if train_loss < best_loss :
                    if accelerator.is_main_process: # 只有主進程保存
                        save_path = os.path.join(config.output_dir, f"checkpoint-best")
                        os.makedirs(save_path, exist_ok=True)

                        # 解包模型以獲取原始 PEFT 模型
                        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                        # 保存 LoRA adapter 權重
                        lora_save_path = os.path.join(save_path, "controlnet_lora")
                        unwrapped_controlnet.save_pretrained(lora_save_path)

                        print(f"\nSaved LoRA checkpoint at step {global_step} to {lora_save_path}")

                        # 可選：如果需要，可以保存優化器和排程器狀態以供恢復訓練
                        # accelerator.save_state(save_path)

                # --- 執行驗證 (如果設定了步數) ---
                # validation_steps = getattr(config, 'validation_steps', None)
                # if validation_steps and global_step % validation_steps == 0:
                    if accelerator.is_main_process: # 只有主進程執行驗證和記錄
                        print(f"\nRunning validation at step {global_step}...")
                        try:
                            # --- 準備驗證 ---
                            # 獲取最新的 LoRA 權重
                            controlnet_for_infer = copy.deepcopy(controlnet)

                            # Merge 用於驗證的那一份
                            if isinstance(controlnet_for_infer, PeftModel):
                                print("Merging LoRA into base model (copy) for validation...")
                                controlnet_for_infer = controlnet_for_infer.merge_and_unload()

                            # 用這份做驗證
                            pipeline.controlnet = controlnet_for_infer
                            
                            # 確保 pipeline 在正確的設備上
                            pipeline.to(accelerator.device)
                            # 設定為評估模式
                            pipeline.controlnet.eval()
                            # pipeline.unet.eval() # UNet 始終是 eval 狀態

                            # --- 從資料集中選取樣本進行驗證 --- # <-- MODIFICATION: Changed logic
                            num_val_samples = getattr(config, 'num_validation_images', 4) # 從設定讀取，預設 4
                            # Clamp num_val_samples to dataset size if necessary
                            num_val_samples = min(num_val_samples, len(train_dataset))

                            val_condition_images = []
                            val_prompts = []
                            val_original_indices = [] # <-- MODIFICATION: Store original indices for logging/saving

                            # Ensure we have a dataset to sample from
                            if len(train_dataset) > 0:
                                # Use random indices for variety each time
                                val_indices = torch.randperm(len(train_dataset))[:num_val_samples].tolist()

                                print(f"Sampling validation data from dataset indices: {val_indices}")

                                for idx in val_indices:
                                    try:
                                        # Get the full sample dictionary from the dataset
                                        sample = train_dataset[idx]

                                        # --- Get Condition Image ---
                                        condition_img_tensor = sample["condition_images"]
                                        # Preprocess: ensure tensor, correct dtype, scale, unsqueeze batch dim
                                        if isinstance(condition_img_tensor, torch.Tensor):
                                            condition_img = condition_img_tensor.to(dtype=weight_dtype).div(255.0).unsqueeze(0)
                                            val_condition_images.append(condition_img.to(accelerator.device))
                                        else:
                                            print(f"Warning: Condition image at index {idx} is not a tensor. Skipping this sample.")
                                            continue # Skip if condition image is not valid

                                        # --- Get Prompt ---
                                        prompt = sample["prompts"] 
                                        # Basic check: Ensure it's a string
                                        if not isinstance(prompt, str):
                                            print(f"Warning: Prompt at index {idx} is not a string ('{prompt}'). Using empty string.")
                                            prompt = "" # Use empty string as fallback
                                        val_prompts.append(prompt)

                                        # Store the original index
                                        val_original_indices.append(idx)

                                    except Exception as e:
                                        print(f"Warning: Error retrieving validation sample at index {idx}: {e}. Skipping.")
                            else:
                                print("Warning: train_dataset is empty. Cannot perform validation.")
                        
                            # --- 生成驗證圖像 ---
                            if val_condition_images and val_prompts: # Check if we have valid pairs
                                validation_images_log = []

                                # <-- MODIFICATION: Iterate through the collected pairs -->
                                for i, (prompt, condition_image_val) in enumerate(zip(val_prompts, val_condition_images)):
                                    original_idx = val_original_indices[i] # Get the original index
                                    print(f"  Generating validation image {i+1}/{len(val_prompts)} using prompt: '{prompt}' (from index {original_idx})")
                                    print(f"  Condition image shape: {condition_image_val.shape}")  
                                    print(f"  Condition image dtype: {condition_image_val.dtype}")
                               
                                    with torch.no_grad():
                                        try: # Add try-except around pipeline call
                                            val_image = pipeline(
                                                prompt=prompt, # Use the prompt from the dataset sample
                                                image=condition_image_val, # Use the condition image from the dataset sample
                                                num_inference_steps=config.num_inference_steps,
                                                guidance_scale=config.guidance_scale,
                                                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
                                            ).images[0]
                                            print(f"  Validation image generated successfully.")
                                            # 記錄到 WandB (如果啟用)
                                            if config.log_wandb:
                                                validation_images_log.append(wandb.Image(
                                                    val_image,
                                                    caption=f"Step {global_step} [Dataset Idx {original_idx}]: {prompt}"
                                                ))

                                            # 可選：保存驗證圖像到本地文件
                                            # <-- MODIFICATION: Use original index in filename -->
                                            val_img_save_path = os.path.join(config.output_dir, f"validation_step_{global_step}_idx_{original_idx}.png")
                                            print(f"  Saving validation image to {val_img_save_path}")
                                            val_image.save(val_img_save_path)
                                        except Exception as pipe_e:
                                            print(f"  Error generating validation image for index {original_idx}: {type(pipe_e).__name__} - {pipe_e}")
                                            print("  🔍 ControlNet type:", type(pipeline.controlnet))
                                            traceback.print_exc()  # 🔍 印出完整錯誤堆疊

                                # --- 記錄到 WandB ---
                                if config.log_wandb and validation_images_log:
                                    accelerator.log({"validation_samples": validation_images_log}, step=global_step)
                                    print(f"Logged {len(validation_images_log)} validation images to WandB.")
                                elif not config.log_wandb and val_prompts: # Check if validation ran
                                    print(f"Generated {len(val_prompts)} validation images locally.")
                            else:
                                print("Skipping validation image generation as no valid data pairs were found.")

                        except Exception as e:
                            print(f"Error during validation setup or sampling: {e}")
                            traceback.print_exc() # Print detailed traceback for debugging
                        finally:
                            # --- 恢復訓練狀態 ---
                            # Make sure controlnet is back on the correct device if moved during validation
                            pipeline.controlnet.to(accelerator.device)
                            pipeline.controlnet.train() # 將 ControlNet 設置回訓練模式
                            print("Validation complete. Resuming training...")


            # --- 檢查是否達到最大步數 ---
            if global_step >= config.max_train_steps:
                break # 跳出內部迴圈 (step loop)

        # 在每個 epoch 結束時記錄平均訓練損失
        epoch_avg_loss = train_loss / num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0.0
        if config.log_wandb:
            accelerator.log({"train_loss_epoch": epoch_avg_loss, "epoch": epoch}, step=global_step)
        print(f"Epoch {epoch+1}/{num_train_epochs} finished. Average Loss: {epoch_avg_loss:.6f}")


        # --- 檢查是否達到最大步數 ---
        if global_step >= config.max_train_steps:
            print(f"\nReached max_train_steps ({config.max_train_steps}). Stopping training.")
            break # 跳出外部迴圈 (epoch loop)

    # --- 訓練結束 ---
    accelerator.wait_for_everyone() # 等待所有進程完成

    # --- 保存最終的 LoRA 模型 ---
    if accelerator.is_main_process:
        final_save_path = os.path.join(config.output_dir, "final-lora-model")
        os.makedirs(final_save_path, exist_ok=True)

        # 解包模型並保存 LoRA 權重
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        lora_final_save_path = os.path.join(final_save_path, "controlnet_lora")
        unwrapped_controlnet.save_pretrained(lora_final_save_path)
        print(f"\nFinal LoRA weights saved to {lora_final_save_path}")

        # 可選：保存完整的 pipeline 狀態（包含 LoRA 權重）
        # pipeline.controlnet = unwrapped_controlnet # 確保 pipeline 使用最終權重
        # pipeline.save_pretrained(final_save_path)
        # print(f"Final pipeline state saved to {final_save_path}")


    # --- 結束 Accelerator 和 WandB ---
    accelerator.end_training()
    if config.log_wandb:
        wandb.finish()

    print("\n--- LoRA Training Completed Successfully! ---")

if __name__ == "__main__":
    main()