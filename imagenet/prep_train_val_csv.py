import os

import paths

with open(paths.wnids_txt) as lines:
    ids = list(map(lambda line: line[:-1], lines))

with open(paths.val_anno_txt) as lines, open(paths.val_csv, 'w') as out:
    for line in lines:
        img_name, img_class, _ = line.split('\t', maxsplit=2)
        img_path = os.path.join(paths.val_dir, 'images', img_name)
        out.writelines("%s %s\n" % (img_path, ids.index(img_class)))


with open(paths.train_csv, 'w') as out:
    for img_class in os.listdir(paths.train_dir):
        with open(os.path.join(paths.train_dir, img_class, "%s_boxes.txt" % img_class)) as lines:
            img_dir = os.path.join(paths.train_dir, img_class, 'images')
            for line in lines:
                img_path = os.path.join(img_dir, line.split('\t', maxsplit=1)[0])
                out.writelines("%s %s\n" % (img_path, ids.index(img_class)))

