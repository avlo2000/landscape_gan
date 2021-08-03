from skimage import io
from skimage.transform import resize
from skimage.filters import sobel
import glob
from tqdm import tqdm
import numpy as np


class DataLoader:
    def __init__(self, img_shape=(512, 512)):
        self._images_list = []
        self.img_shape = img_shape
        self._data_x = None
        self._data_y = None

    def load(self, images: int = -1):
        print('Loading dataset...')
        for filename in tqdm(glob.glob('data/landscape/*.jpg')[:images]):
            img = io.imread(filename)
            img = resize(img, self.img_shape)
            self._images_list.append(img)
        self.img_shape = self._images_list[0].shape

    def prepare(self, data_type='noised', transpose=False):
        if data_type == 'identity':
            self._data_x = self._images_list
            self._data_y = self._images_list
        if data_type == 'noised':
            self._data_y = self._images_list
            self._data_x = []
            for img in self._images_list:
                self._data_x.append(sobel(img))
        if transpose:
            for i in range(len(self._data_x)):
                self._data_x[i] = np.transpose(self._data_x[i], (2, 0, 1))
                self._data_y[i] = np.transpose(self._data_y[i], (2, 0, 1))

    def show_first(self, num=3):
        for i in range(num):
            io.imshow(self._data_x[i])
            io.show()
            io.imshow(self._data_y[i])
            io.show()

    def get_data(self, split=0.8):
        sz = len(self._images_list)
        index = int(sz*split)
        train_x = self._data_x[:index-1]
        test_x = self._data_x[index:]

        train_y = self._data_y[:index-1]
        test_y = self._data_y[index:]

        return train_x, test_x, train_y, test_y
