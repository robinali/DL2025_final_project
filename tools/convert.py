import os
from PIL import Image

# Directory containing the .jpg files
input_dir = "./fake"  # Change this to your directory path

# Convert all .jpg or .jpeg files to .png
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg")):
        jpg_path = os.path.join(input_dir, filename)
        png_path = os.path.join(input_dir, os.path.splitext(filename)[0] + ".png")

        # Open and convert image
        with Image.open(jpg_path) as img:
            img.convert("RGB").save(png_path, "PNG")
            print(f"Converted: {filename} -> {os.path.basename(png_path)}")

