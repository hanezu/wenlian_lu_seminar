import os, paths

import cv2

def check_images(csv):
    with open(csv) as lines:
        for line in lines:
            img_path, img_class = line.split(' ')
            img = cv2.imread(img_path)
            if img.shape[2] != 3:
                print("not 3 channel!")

check_images(paths.train_csv)
check_images(paths.val_csv)
