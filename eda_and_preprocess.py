import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "processed_images"
EDA_FOLDER = "eda_outputs"
SIZE = 224

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EDA_FOLDER, exist_ok=True)

# -------------------------------
# TRACKERS
# -------------------------------
formats = []
widths = []
heights = []
corrupt = 0

# -------------------------------
# PROCESS FUNCTION
# -------------------------------
def process_image(img):
    img = img.convert("RGB")

    # keep aspect ratio
    img.thumbnail((SIZE, SIZE))

    # pad to square
    new_img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))

    x = (SIZE - img.size[0]) // 2
    y = (SIZE - img.size[1]) // 2

    new_img.paste(img, (x, y))
    return new_img

# -------------------------------
# MAIN LOOP
# -------------------------------
files = os.listdir(INPUT_FOLDER)
print(f"Total files found: {len(files)}")

count = 0

for file in tqdm(files, desc="Processing images"):
    path = os.path.join(INPUT_FOLDER, file)

    try:
        img = Image.open(path)

        # ---- EDA ----
        formats.append(img.format)
        w, h = img.size
        widths.append(w)
        heights.append(h)

        # ---- PREPROCESS ----
        processed = process_image(img)

        save_path = os.path.join(OUTPUT_FOLDER, f"{count}.png")
        processed.save(save_path, "PNG")

        count += 1

    except:
        corrupt += 1
        continue

# -------------------------------
# SUMMARY
# -------------------------------
print("\n===== SUMMARY =====")
print(f"Processed: {count}")
print(f"Corrupt skipped: {corrupt}")

# -------------------------------
# EDA PLOTS
# -------------------------------
print("\nGenerating EDA plots...")

# Format distribution
format_counts = Counter(formats)
plt.figure()
plt.bar(format_counts.keys(), format_counts.values())
plt.title("Image Format Distribution")
plt.savefig(os.path.join(EDA_FOLDER, "formats.png"))
plt.close()

# Width distribution
plt.figure()
plt.hist(widths, bins=30)
plt.title("Width Distribution")
plt.savefig(os.path.join(EDA_FOLDER, "widths.png"))
plt.close()

# Height distribution
plt.figure()
plt.hist(heights, bins=30)
plt.title("Height Distribution")
plt.savefig(os.path.join(EDA_FOLDER, "heights.png"))
plt.close()

print("EDA saved to:", EDA_FOLDER)
print("Processed images saved to:", OUTPUT_FOLDER)