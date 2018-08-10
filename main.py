import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from image_registration import alignImages


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


class Main:
    def __init__(self, images_dir='images'):
        self.data_handler = DataHandler(images_dir)

    def run(self):
        for reference_image, inspection_image in self.data_handler.get():
            detector = DefectDetector(reference_image, inspection_image)
            detector.run()


class DefectDetector:
    def __init__(self, reference_image, inspection_image):
        self.reference_image = reference_image
        self.inspection_image = inspection_image

    def run(self):
        inspection_image_registered, h, matches_image = alignImages(self.inspection_image, self.reference_image)
        diff = cv2.absdiff(inspection_image_registered, self.reference_image)
        plt.imshow(diff)
        # plt.imshow(np.concatenate((inspection_image_registered, self.reference_image), axis=1))
        plt.show()
        pass


def main():
    Main().run()


if __name__ == '__main__':
    main()
