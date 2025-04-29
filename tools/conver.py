import os
from PIL import Image

# Increase the pixel limit for decompression bombs (example: 500 million pixels)
Image.MAX_IMAGE_PIXELS = 500000000000

# Specify the directory containing the TIFF files
input_directory = './group'  # Replace with the path to your directory
output_directory = './jpg'  # Replace with the desired output directory

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through all the files in the directory
for filename in os.listdir(input_directory):
    # Check if the file is a .tif file
    if filename.lower().endswith('.tif'):
        # Construct the full file path
        tif_path = os.path.join(input_directory, filename)

        try:
            # Open the TIFF image
            with Image.open(tif_path) as tif_image:
                # Convert the TIFF image to RGB and save it as a JPG
                jpg_filename = os.path.splitext(filename)[0] + '.jpg'
                jpg_path = os.path.join(output_directory, jpg_filename)
                tif_image.convert("RGB").save(jpg_path, "JPEG")
            print(f"Converted {filename} to {jpg_filename}")

        except:
            # If there's an error (corrupted file), print it and skip the file
            print(f"Error processing {filename}")

