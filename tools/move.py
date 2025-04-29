import os
import shutil

# Specify the root directory and the new destination directory
source_directory = './'  # Replace with your source directory
destination_directory = './group'  # Replace with your destination directory

# Create destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Walk through the source directory and its subdirectories
for root, dirs, files in os.walk(source_directory):
    for filename in files:
        # Check if the file is a .tif or .tiff file
        if filename.lower().endswith(('.tif', '.tiff')):
            # Construct full file path
            source_file = os.path.join(root, filename)
            
            # Construct the destination path
            destination_file = os.path.join(destination_directory, filename)
            
            # Copy the file to the destination directory
            shutil.copy2(source_file, destination_file)
            print(f"Copied {filename} to {destination_directory}")

