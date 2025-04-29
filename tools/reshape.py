import os
from PIL import Image

def zoom_out_width_and_resize(img, final_size=256):
    width, height = img.size

    # Shrink width to match height (zoom out)
    new_width = height
    img = img.resize((new_width, height), Image.LANCZOS)

    # Paste the image into a square canvas (pad top and bottom)
    square_img = Image.new("RGB", (new_width, new_width), (0, 0, 0))  # black padding
    offset_y = (new_width - height) // 2
    square_img.paste(img, (0, offset_y))

    # Final resize to 256x256
    return square_img.resize((final_size, final_size), Image.LANCZOS)

def process_directory(input_dir, output_dir, final_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    img = zoom_out_width_and_resize(img, final_size)
                    img.save(output_path)
                    print(f"Processed: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Example usage
input_directory = "input_images"
output_directory = "output_images"
process_directory(input_directory, output_directory)

