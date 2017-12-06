import cv2
from chainer.links.caffe import CaffeFunction

import numpy as np
from wiki.VGG import VGG16Layers

vgg = VGG16Layers('dex_imdb_wiki.npz')
# vgg = CaffeFunction('dex_imdb_wiki.caffemodel')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    w, h, ch = frame.shape
    frame = frame[:, h // 2 - w // 2: h // 2 + w // 2:]

    input_img = cv2.resize(frame, (224, 224))

    input_img = input_img.transpose((2, 0, 1))[np.newaxis].astype('f')
    # input_img: np.ndarray (1, ch, 224, 224)

    res = vgg(input_img)
    prob = res['prob'].data[0]
    real_age = prob.dot(np.arange(101))
    cv2.imshow("real_age", frame)
    print(int(real_age))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
