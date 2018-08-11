import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation
import cv2


class TranslationTransform:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def transform(self, image):
        return scipy.ndimage.interpolation.shift(image, [self.dy, self.dx])

    def get_valid_mask(self, image_shape):
        dx, dy = self.dx, self.dy
        mask = np.ones(image_shape, dtype=np.bool)
        if dy > 0:
            dy = np.int(np.ceil(dy))
            mask[:dy, :] = False
        else:
            dy = np.int(np.floor(dy))
            mask[dy:, :] = False
        if dx > 0:
            dx = np.int(np.ceil(dx))
            mask[:, :dx] = False
        else:
            dx = np.int(np.floor(dx))
            mask[:, dx:] = False
        return mask


def list_dir_recursive(d):
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(d) for f in fn]


class DataHandler:
    def __init__(self, images_dir):
        self.images_dir = images_dir
        file_paths = list_dir_recursive(self.images_dir)
        image_paths = [p for p in file_paths if '.tif' in p]
        self.gt_path = list(set(file_paths) - set(image_paths))[0]
        reference_images = [p for p in image_paths if 'reference' in p]
        inspected_images = [p.replace('reference', 'inspected') for p in reference_images]
        assert set(inspected_images).issubset(set(image_paths))
        self.image_pairs = list(zip(reference_images, inspected_images))

    def get(self):
        for image_pair in self.image_pairs:
            yield [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_pair]


def show_image(image, title=''):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.show()
