import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import ast
from final_project.src.utils import Config, ArtemisDataset
from torchvision import transforms
import torch
import os


# ==========================================
# 5. INFERENCE HELPERS
# ==========================================

def rebuild_vocab_mapping(dataset):
    """
    Since we don't have the tokenizer file, we reconstruct the ID->Word map
    by scanning the DataFrame.
    """
    print("Reconstructing vocabulary mapping from dataset...")
    idx_to_word = {}

    if 'tokens' not in dataset.df.columns:
        print("Warning: 'tokens' column missing. Cannot reconstruct words. Output will be indices.")
        return {}

    scan_limit = min(len(dataset.df), 10000)

    for i in tqdm(range(scan_limit), desc="Building Vocab"):
        row = dataset.df.iloc[i]
        tokens = row['tokens']

        # --- KEY FIX: PARSE STRING "['word', 'word']" TO LIST ---
        if isinstance(tokens, str):
            try:
                tokens = ast.literal_eval(tokens)
            except (ValueError, SyntaxError):
                continue

        indices = row['tokens_encoded'] # List of ints [1, 4, 5, ...]

        # Mapping Logic:
        # tokens_encoded usually = [SOS, word1_id, word2_id, ... EOS, PAD...]
        # tokens = [word1, word2, ...]
        # So indices[1] corresponds to tokens[0]

        current_token_idx = 0
        current_vocab_idx = 1 # Skip SOS

        while current_token_idx < len(tokens) and current_vocab_idx < len(indices):
            idx = indices[current_vocab_idx]

            if idx in [Config.PAD_IDX, Config.EOS_IDX]:
                break

            word = tokens[current_token_idx]

            if idx not in idx_to_word:
                idx_to_word[idx] = word

            current_token_idx += 1
            current_vocab_idx += 1

    return idx_to_word

def generate_caption_beam_search(model, image_path, vocab_size, idx_to_word=None, beam_width=5, max_len=20, repetition_penalty=1.5):
    model.eval()

    # Process Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    except Exception as e:
        print(f"Error loading image: {e}")
        return ""

    with torch.no_grad():
        # 1. Encode Image
        encoder_out = model.encoder(image_tensor)
        encoder_out = encoder_out.expand(beam_width, -1, -1)

        # 2. Initialize Beams
        top_k_seqs = torch.tensor([[Config.SOS_IDX]] * beam_width, dtype=torch.long).to(Config.DEVICE)
        top_k_scores = torch.zeros(beam_width).to(Config.DEVICE)
        top_k_scores[1:] = float('-inf')

        # 3. Beam Search Loop
        completed_seqs = []

        for step in range(max_len):
            decoder_out = model.decoder(encoder_out, top_k_seqs)
            next_token_logits = decoder_out[:, -1, :]

            # --- APPLY REPETITION PENALTY ---
            # Penalize tokens that have already appeared in the sequence
            if repetition_penalty > 1.0:
                for i in range(beam_width):
                    for prev_token in top_k_seqs[i]:
                        val = prev_token.item()
                        if val in [Config.SOS_IDX, Config.PAD_IDX]:
                            continue

                        # If logit is negative (usual), multiply to make it smaller (more negative)
                        # If logit is positive, divide to make it smaller
                        if next_token_logits[i, val] < 0:
                            next_token_logits[i, val] *= repetition_penalty
                        else:
                            next_token_logits[i, val] /= repetition_penalty

            next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)
            current_scores = top_k_scores.unsqueeze(1) + next_token_logprobs

            flat_scores = current_scores.view(-1)
            best_scores, best_indices = torch.topk(flat_scores, beam_width)

            beam_indices = best_indices // vocab_size
            token_indices = best_indices % vocab_size

            next_top_k_seqs_list = []
            new_scores = []

            for i in range(beam_width):
                b_idx = beam_indices[i]
                t_idx = token_indices[i]
                score = best_scores[i]

                prev_seq = top_k_seqs[b_idx]

                if t_idx.item() == Config.EOS_IDX:
                    completed_seqs.append((score.item(), prev_seq.tolist()))

                new_seq = torch.cat([prev_seq, torch.tensor([t_idx], device=Config.DEVICE)])
                next_top_k_seqs_list.append(new_seq)
                new_scores.append(score)

            top_k_seqs = torch.stack(next_top_k_seqs_list)
            top_k_scores = torch.tensor(new_scores).to(Config.DEVICE)

            if len(completed_seqs) >= beam_width:
                break

        if len(completed_seqs) > 0:
            completed_seqs.sort(key=lambda x: x[0], reverse=True)
            best_seq = completed_seqs[0][1]
        else:
            best_seq = top_k_seqs[0].tolist()

    # Convert to words
    if idx_to_word:
        words = []
        for idx in best_seq:
            if idx not in [Config.SOS_IDX, Config.EOS_IDX, Config.PAD_IDX]:
                words.append(idx_to_word.get(idx, str(idx)))
        return " ".join(words)
    else:
        return f"Indices: {best_seq}"


def run_inference(model):
    # 1. Setup
    if not model:
        model_path = os.path.join(Config.ROOT_DIR, "artemis_captioner_resnet_transformer.pth")
        print(f"No model passed, loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        if not os.path.exists(model_path):
            print("Model file not found. Please train first.")
            return

    # 2. Load Dataset
    dataset = ArtemisDataset(Config.PICKLE_PATH, split='train')
    vocab_size = dataset.vocab_size

    # Rebuild vocab map
    idx_to_word = rebuild_vocab_mapping(dataset)

    for i in range(5):
        # 4. Generate for a random image
        print("\n--- Generating Example Caption (Beam Search) ---")
        rand_idx = random.randint(0, len(dataset))
        row = dataset.df.iloc[rand_idx]
        img_path = row['img_path']

        print(f"Image Path: {img_path}")

        caption = generate_caption_beam_search(model, img_path, vocab_size, idx_to_word=idx_to_word, beam_width=5)

        # 5. Display
        try:
            img = Image.open(img_path).convert("RGB")
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Generated: {caption}", fontsize=12, wrap=True)
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

        print(f"Generated Caption: {caption}")
