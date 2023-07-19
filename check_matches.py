import os
import shutil

def compare_folders(folder1, folder2, output_folder):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    unique_files = files1.symmetric_difference(files2)
    
    for file in unique_files:
        base_name = os.path.splitext(file)[0]
        matching_file = [f for f in unique_files if os.path.splitext(f)[0] == base_name]
        
        if len(matching_file) == 1:
            matching_file = matching_file[0]
            
            if file in files1:
                shutil.move(os.path.join(folder1, file), output_folder)
            else:
                shutil.move(os.path.join(folder2, file), output_folder)

# Specify the folder paths
folder1_path = "./segmentation/"
folder2_path = "./original/"
output_folder_path = "./no_matches/"

# Call the function to compare and move files
compare_folders(folder1_path, folder2_path, output_folder_path)
