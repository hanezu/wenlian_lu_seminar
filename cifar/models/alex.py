import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 32
    # hidden_num = 4096
    # hidden_num = 256
    hidden_num = 2048

    def __init__(self, class_labels=10):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 7, stride=2, pad=3)  # original k=11, pad=0
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, self.hidden_num)
            self.fc7 = L.Linear(None, self.hidden_num)
            self.fc8 = L.Linear(None, class_labels)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        # loss = F.softmax_cross_entropy(h, t)
        # chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return h

