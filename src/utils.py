import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import ast


# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Colab Paths matching your structure
    ROOT_DIR = "/Users/alexis/Desktop/Desktop/OMSCS/Fall2025/CS7643/final_project"
    PICKLE_PATH = os.path.join(ROOT_DIR, "processed_artemis/artemis_image_caption_dataset.pkl")
    MINI_PICKLE_PATH = os.path.join(ROOT_DIR, "processed_artemis/artemis_image_caption_dataset_mini.pkl")

    # Model Hyperparameters
    EMBED_DIM = 512
    HIDDEN_DIM = 512
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Training Settings
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Vocabulary (Placeholder, overwritten by Dataset)
    VOCAB_SIZE = 30000
    PAD_IDX = 0
    SOS_IDX = 1  # Standard for Artemis (based on data inspection)
    EOS_IDX = 2  # Standard for Artemis (based on data inspection)


# ==========================================
# 2. DATASET
# ==========================================
class ArtemisDataset(Dataset):
    def __init__(self, pickle_path, transform=None, split='train'):
        print(f"Loading dataset from {pickle_path}...")
        self.df = pd.read_pickle(pickle_path)

        # Filter by split if the column exists
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.transform = transform

        # --- PRE-PROCESSING & VOCAB SIZE DETECTION ---
        # 1. Ensure tokens are lists (parse strings if necessary)
        if len(self.df) > 0 and isinstance(self.df.iloc[0]['tokens_encoded'], str):
            print("Parsing token strings to lists...")
            tqdm.pandas(desc="Parsing tokens")
            self.df['tokens_encoded'] = self.df['tokens_encoded'].apply(ast.literal_eval)

        # 2. Calculate Vocab Size dynamically
        print("Calculating vocabulary size from dataset...")
        max_token = self.df['tokens_encoded'].explode().max()

        if pd.isna(max_token):
            self.vocab_size = 30000
        else:
            self.vocab_size = int(max_token) + 1

        print(f"Detected Max Token Index: {max_token}")
        print(f"Setting VOCAB_SIZE to: {self.vocab_size}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        img_path = row['img_path']
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # 2. Load Caption
        caption_seq = row['tokens_encoded']
        caption_tensor = torch.tensor(caption_seq, dtype=torch.long)

        # --- SAFETY CLAMP ---
        # This prevents CUDA device-side asserts by ensuring no token exceeds the vocab limit.
        # It handles rare edge cases or data inconsistencies.
        caption_tensor = torch.clamp(caption_tensor, min=0, max=self.vocab_size - 1)

        return image, caption_tensor
