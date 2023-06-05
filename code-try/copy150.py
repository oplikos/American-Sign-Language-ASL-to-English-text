import os
import shutil

source_directory = 'C:/Users/OPLIK/OneDrive/Desktop/SCHOOL/UCSD/Spring 2023/CSE 145/ASL-Translator/Data/combienData'

# Directory to copy the selected images
destination_directory = 'C:/Users/OPLIK/OneDrive/Desktop/SCHOOL/UCSD/Spring 2023/CSE 145/ASL-Translator/Data/myDataRe2'

num_images_per_folder = 150  # Number of images to copy from each folder

# Get a list of all folders in the source directory
folders = [folder for folder in os.listdir(source_directory) if os.path.isdir(
    os.path.join(source_directory, folder))]

# Iterate over each folder
for folder in folders:
    source_folder = os.path.join(source_directory, folder)
    destination_folder = os.path.join(destination_directory, folder)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all images in the source folder
    images = [image for image in os.listdir(
        source_folder) if os.path.isfile(os.path.join(source_folder, image))]

    # Copy the specified number of images to the destination folder
    selected_images = images[:num_images_per_folder]
    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.copy2(source_path, destination_path)
