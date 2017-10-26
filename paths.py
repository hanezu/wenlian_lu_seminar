import os

dataset_dir = "../datasets"
imagenet_dir = os.path.join(dataset_dir, 'tiny-imagenet-200')

wnids_txt = os.path.join(imagenet_dir, 'wnids.txt')

val_dir = os.path.join(imagenet_dir, 'val')
train_dir = os.path.join(imagenet_dir, 'train')
val_anno_txt = os.path.join(val_dir, 'val_annotations.txt')

val_csv = os.path.join(imagenet_dir, 'val.csv')
train_csv = os.path.join(imagenet_dir, 'train.csv')