import nltk
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast

# !python artemis/scripts/preprocess_artemis_data.py \
#     -save-out-dir ./processed_artemis \
#     -raw-artemis-data-csv ./official_data/artemis_dataset_release_v0.csv \
#     --preprocess-for-deep-nets True



ROOT = "/Users/alexis/Desktop/Desktop/OMSCS/Fall2025/CS7643/final_project"
PKL_PATH = f"{ROOT}/processed_artemis/artemis_image_caption_dataset.pkl"
CSV_PATH = f"{ROOT}/processed_artemis/artemis_preprocessed.csv"
IMG_ROOT = f"{ROOT}/images"   # has expressionism / *.jpg

df = pd.read_csv(CSV_PATH)
df.head()

nltk.download('punkt')
nltk.download('punkt_tab')

records = []
for style in os.listdir(IMG_ROOT):
    style_dir = os.path.join(IMG_ROOT, style)
    if not os.path.isdir(style_dir):
        continue

    # for artist in os.listdir(style_dir):
    #     artist_dir = os.path.join(style_dir, artist)
    #     if not os.path.isdir(artist_dir):
    #         continue

    for fname in os.listdir(style_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        slug = os.path.splitext(fname)[0]
        # painting_id = f"{artist}_{slug}"
        img_path = os.path.join(style_dir, fname)
        records.append((slug, style.lower(), img_path))

img_df = pd.DataFrame(records, columns=["painting", "art_style", "img_path"])

# Normalize text case to guarantee join matches
df["painting"] = df["painting"].astype(str)
df["art_style"] = df["art_style"].str.lower()
img_df["art_style"] = img_df["art_style"].str.lower()

linked = df.merge(img_df, on=["painting", "art_style"], how="inner")
print("Images found:", len(img_df))
print("Matched rows:", len(linked))

linked['tokens_encoded'] = linked['tokens_encoded'].apply(ast.literal_eval)

# Save for future use
linked.to_pickle(PKL_PATH)

print("Saved to:", PKL_PATH)

linked = pd.read_pickle(PKL_PATH)
linked.head()

# Quick visual check
sample = linked.sample(4)

fig, axes = plt.subplots(2, 2, figsize=(10,10))
for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
    img = Image.open(row["img_path"]).convert("RGB")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{row['emotion']}:\n{row['utterance_spelled']}", fontsize=8)
plt.tight_layout()
plt.show()
