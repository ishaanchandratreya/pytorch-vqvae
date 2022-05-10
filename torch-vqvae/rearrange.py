import os
import shutil

base_path = '/proj/vondrick3/ishaan/old/mini-imagenet'
for path in ['train','val', 'test']:

    source_path = os.path.join(base_path, path)
    copy_path = os.path.join(base_path, f'{path}_final', 'images')
    path_items = os.listdir(source_path)
    for item_folder in path_items:
        if 'n' in item_folder:

            item_folder_path = os.path.join(source_path, item_folder)
            item_folder_ims = os.listdir(item_folder_path)
            for im in item_folder_ims:
                if '.jpg' in im:
                    shutil.copyfile(os.path.join(item_folder_path, im), os.path.join(copy_path, im))