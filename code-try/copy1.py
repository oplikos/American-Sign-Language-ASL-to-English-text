import os
import random
import shutil

def copy_files(source_folder, destination_folder):
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            num_files_to_copy = len(files) // 2  # 50% of the files
            files_to_copy = random.sample(files, num_files_to_copy)
            
            new_folder_path = os.path.join(destination_folder, folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            for file_name in files_to_copy:
                file_path = os.path.join(folder_path, file_name)
                shutil.copy(file_path, new_folder_path)
                
# Example usage
source_folder = 'newData'
destination_folder = 'newData50'

copy_files(source_folder, destination_folder)
