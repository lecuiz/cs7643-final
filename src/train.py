import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

from final_project.src.utils import Config, ArtemisDataset

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train(model, dataset: ArtemisDataset | None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    if not os.path.exists(Config.PICKLE_PATH):
        print(f"ERROR: File not found at {Config.PICKLE_PATH}")
        return

    if not dataset:
        dataset = ArtemisDataset(Config.PICKLE_PATH, transform=transform, split='train')

    # Reduced num_workers to 0 to make debugging easier if errors persist
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Initializing Model with Vocab Size: {dataset.vocab_size}")
    # model = ImageCaptioningModel(
    #     vocab_size=dataset.vocab_size,
    #     embed_dim=Config.EMBED_DIM,
    #     hidden_dim=Config.HIDDEN_DIM,
    #     num_heads=Config.NUM_HEADS,
    #     num_layers=Config.NUM_LAYERS,
    #     dropout=Config.DROPOUT
    # ).to(Config.DEVICE)

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
