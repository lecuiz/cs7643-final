import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
    DEVICE = "cuda" if torch.cuda.is_available() else "mps"

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

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================

class EncoderCNN(nn.Module):
    """
    The Visual Backbone (ResNet-101)
    """
    def __init__(self, embed_dim):
        super(EncoderCNN, self).__init__()
        # Load pretrained ResNet-101
        try:
            from torchvision.models import ResNet101_Weights
            resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        except (ImportError, AttributeError):
            resnet = models.resnet101(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(2048, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(0, 2, 1)
        features = self.embed(features)
        return features

class DecoderTransformer(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, dropout):
        super(DecoderTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self._generate_positional_encoding(embed_dim, max_len=5000)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def _generate_positional_encoding(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, features, captions):
        embeddings = self.embedding(captions) * torch.sqrt(torch.tensor(self.embed_dim).float().to(captions.device))
        seq_len = captions.size(1)
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :].to(captions.device)
        embeddings = self.dropout(embeddings)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        output = self.transformer_decoder(tgt=embeddings, memory=features, tgt_mask=tgt_mask)
        prediction = self.fc_out(output)
        return prediction

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, dropout):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderTransformer(vocab_size, embed_dim, hidden_dim, num_heads, num_layers, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    if not os.path.exists(Config.PICKLE_PATH):
        print(f"ERROR: File not found at {Config.PICKLE_PATH}")
        return

    dataset = ArtemisDataset(Config.PICKLE_PATH, transform=transform, split='train')

    # Reduced num_workers to 0 to make debugging easier if errors persist
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Initializing Model with Vocab Size: {dataset.vocab_size}")
    model = ImageCaptioningModel(
        vocab_size=dataset.vocab_size,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

    print(f"Starting Training on {Config.DEVICE}...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")

        for images, captions in loop:
            images = images.to(Config.DEVICE)
            captions = captions.to(Config.DEVICE)

            # Safety check for empty batches or sequences
            if captions.size(1) <= 1:
                continue

            decoder_input = captions[:, :-1]
            targets = captions[:, 1:]

            optimizer.zero_grad()

            outputs = model(images, decoder_input)

            loss = criterion(outputs.reshape(-1, dataset.vocab_size), targets.reshape(-1))

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    save_path = os.path.join(Config.ROOT_DIR, "artemis_captioner_resnet_transformer.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
