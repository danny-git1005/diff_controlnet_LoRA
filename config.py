# -*- coding: utf-8 -*-
import torch

class Config:
    # --- 基本路徑和模型設定 ---
    # *必須修改* - 你的基礎 Stable Diffusion 模型 (Hugging Face 名稱或本地路徑)
    base_model: str = r"E:\aerial_img\phase2_diffusion_mdoel\diff_controlnet_LoRA\weight\final-model"
    # *必須修改* - 你的 ControlNet 基礎模型 (Hugging Face 名稱或本地路徑，用於提取架構)
    # 例如: "lllyasviel/sd-controlnet-canny" 或 "lllyasviel/sd-controlnet-depth"
    controlnet_model: str = r"E:\aerial_img\phase2_diffusion_mdoel\diff_controlnet_LoRA\weight\final-model\controlnet"

    data_dir = "E:/aerial_img/phase2_diffusion_mdoel/data/preprocess"
    # *必須修改* - 存放訓練圖像的資料夾路徑
    image_dir: str =  f"{data_dir}/output_flipped_lane_label"
    # *必須修改* - 存放對應條件圖 (Canny邊緣圖, 深度圖等) 的資料夾路徑
    condition_dir: str = f"{data_dir}/output_flipped_lane_label_masks"
    # *必須修改* - 儲存訓練檢查點和最終模型的資料夾路徑
    output_dir: str = "outputs"

    # --- 訓練參數 ---
    # 訓練圖像的目標解析度 (會被 resize 成這個大小)
    resolution: int = 512
    # 學習率
    learning_rate: float = 1e-4
    # 每個 GPU 上的批次大小
    train_batch_size: int = 4 # <--- 根據你的 GPU VRAM 調整
    # 總訓練步數
    max_train_steps: int = 40 # <--- 設定你希望的總訓練步數
    # 梯度累積步數 (實際的總 batch size = train_batch_size * num_gpus * gradient_accumulation_steps)
    gradient_accumulation_steps: int = 4
    # 混合精度訓練 ("no", "fp16", "bf16") - "fp16" 通常是個好選擇，如果 GPU 支持 "bf16" 可能更好
    mixed_precision: str = "no" # <--- 根據你的 GPU 調整

    # --- LoRA 特定參數 ---
    # *必須為 True* 以啟用 LoRA 訓練
    use_lora: bool = True
    # LoRA 矩陣的秩 (rank) - 常見值: 4, 8, 16, 32, 64
    lora_rank: int = 16 # <--- 可以實驗調整
    # LoRA 的 alpha 縮放因子 (通常設為與 rank 相同)
    lora_alpha: int = 16 # <--- 通常與 lora_rank 相同
    # LoRA 層的 dropout 機率 (可選)
    lora_dropout: float = 0.1

    # --- 優化器和排程器 ---
    # AdamW 優化器參數
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    # 學習率排程器類型 ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup")
    lr_scheduler: str = "cosine"
    # 學習率預熱步數
    lr_warmup_steps: int = 500
    # 梯度裁剪的最大範數值 (設為 None 則不裁剪)
    max_grad_norm: float = 1.0

    # --- 資料和條件 ---
    # 是否使用文字提示詞 (如果為 True，需要提供 prompt_file)
    use_text_condition: bool = True
    # *必須修改* (如果 use_text_condition=True) - 包含提示詞的 JSON 文件路徑
    # JSON 格式: {"image_filename_without_extension": "prompt text", ...}
    prompt_file: str = f"{data_dir}/intersection_prompts.json"
    # 條件類型標識符 (用於記錄和可能的未來邏輯，應與 controlnet_model 對應)
    condition_type: str = "None" # 例如: "canny", "depth", "pose", "seg"
    # DataLoader 的工作進程數
    num_workers: int = 4 # <--- 根據你的 CPU 和系統調整

    # --- 記錄和驗證 ---
    # 是否啟用 Weights & Biases 記錄
    log_wandb: bool = True
    # (可選) 你的 WandB API 金鑰 (如果設定了，會覆蓋環境變數)
    wandb_api_key: str | None = None # "YOUR_WANDB_API_KEY"
    # (可選) 你的 WandB 實體名稱 (用戶名或團隊名)
    wandb_entity: str | None = "danny_paper_project" # <--- 修改成你的 WandB 用戶名
    # WandB 專案名稱
    project_name: str = "diff_phase_3070ti" # <--- 修改成你喜歡的專案名
    # 每隔多少步保存一次檢查點 (LoRA 權重)
    save_steps: int = 5
    # 每隔多少步執行一次驗證 (生成樣本圖像) - 設為 None 或 0 則不驗證
    validation_steps: int = 5
    # 驗證時生成的圖像數量
    num_validation_images: int = 4
    # 用於驗證圖像生成的提示詞列表
    # 驗證時推理使用的步數
    num_inference_steps: int = 60
    # 驗證時推理使用的 guidance scale
    guidance_scale: float = 7.5
    # 驗證時 ControlNet 的條件強度
    controlnet_conditioning_scale: float = 1.0

    # --- 其他 ---
    # 隨機種子，用於可重複實驗 (設為 None 則不固定)
    seed: int | None = 42
    # 是否啟用梯度檢查點 (可以節省 VRAM，但可能稍慢) - 主要用於 ControlNet 本身，PEFT 也有自己的實現
    # gradient_checkpointing: bool = False # 在 PEFT 中通常透過 LoraConfig 控制或自動處理

    # --- 類別初始化 (不需要修改) ---
    def __init__(self):
        # 你可以在這裡添加一些動態計算或檢查，如果需要的話
        # 例如: 檢查路徑是否存在
        # if not os.path.exists(self.image_dir):
        #     print(f"Warning: image_dir '{self.image_dir}' does not exist.")
        pass

    # --- 允許像字典一樣訪問屬性 (可選) ---
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

# --- 方便直接運行 config.py 來檢查設定 (可選) ---
if __name__ == "__main__":
    config = Config()
    print("--- Configuration Settings ---")
    for key, value in vars(config).items():
        if not key.startswith("__"): # 不顯示內部屬性
            print(f"{key}: {value}")
    print("----------------------------")