import os
from PIL import Image

# Specify the directory containing the JPG files
input_directory = './images'  # Replace with the path to your directory
output_directory = './resized_and_chopped_images'  # Directory to save resized images

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through all the files in the directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith('.jpg'):
        # Construct the full file path
        jpg_path = os.path.join(input_directory, filename)

        try:
            # Open the JPG image
            with Image.open(jpg_path) as img:
                # Get the original image dimensions
                width, height = img.size

                # Resize the image based on the shorter edge
                if width < height:
                    new_width = 512
                    new_height = int((height / width) * 512)
                else:
                    new_height = 512
                    new_width = int((width / height) * 512)

                # Resize the image while maintaining the aspect ratio
                img = img.resize((new_width, new_height), Image.ANTIALIAS)

                # Crop the image to 512x512 based on the longer edge
                left = (new_width - 512) // 2
                top = (new_height - 512) // 2
                right = (new_width + 512) // 2
                bottom = (new_height + 512) // 2

                # Perform the crop
                img = img.crop((left, top, right, bottom))

                # Save the resized and cropped image
                resized_cropped_filename = os.path.splitext(filename)[0] + '_resized_chopped.jpg'
                resized_cropped_path = os.path.join(output_directory, resized_cropped_filename)
                img.save(resized_cropped_path)

            print(f"Resized and chopped {filename} and saved as {resized_cropped_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

