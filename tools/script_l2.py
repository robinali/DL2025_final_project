from PIL import Image
import os
import sys


def convert_to_jpg(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # Ensure it's in RGB mode

            # Crop to the largest square possible
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img = img.crop((left, top, right, bottom))

            # Resize to 512x512
            img = img.resize((512, 512), Image.ANTIALIAS)

            img.save(output_path, "JPEG")
            print(f"Converted and saved: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error converting {os.path.abspath(input_path)}: {e}")


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
            convert_to_jpg(input_path, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(input_dir, output_dir)
