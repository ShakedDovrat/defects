import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import apply_hysteresis_threshold

from image_registration import ImageAligner


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
            detector = DefectDetector(reference_image, inspection_image, debug=True)
            detector.run()


def relative_diff(a, b):
    return np.maximum(a, b) / np.minimum(a, b)


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


class DefectDetector:
    MEDIAN_FILTER_SIZE = 3
    # LOW_DIFF_THRESHOLD = 1.2
    # HIGH_DIFF_THRESHOLD = 1.7
    LOW_DIFF_THRESHOLD = 70
    HIGH_DIFF_THRESHOLD = 110
    MORPHOLOGY_SE_SIZE = (3, 3)

    def __init__(self, reference_image, inspection_image, debug=False):
        self.reference_image = reference_image
        self.inspection_image = inspection_image
        self.debug = debug
        self.reference_image_registered = None
        self.valid_registration_mask = None

    def run(self):
        self._pre_process()
        self._register()
        valid_diff_mask = self._diff()
        output_mask = self._post_process(valid_diff_mask)
        # if self.debug:
        show_image(output_mask)
        plt.close('all')

    def _pre_process(self):
        # if self.debug:
        show_image(self.inspection_image)
        cv2.medianBlur(self.reference_image, self.MEDIAN_FILTER_SIZE, self.reference_image)
        cv2.medianBlur(self.inspection_image, self.MEDIAN_FILTER_SIZE, self.inspection_image)
        # GAUSS_FILTER_SIZE = (0, 0)  # 0 means compute from sigma
        # GAUSS_SIGMA = 3
        # cv2.GaussianBlur(self.reference_image, GAUSS_FILTER_SIZE, GAUSS_SIGMA, self.reference_image)
        # cv2.GaussianBlur(self.inspection_image, GAUSS_FILTER_SIZE, GAUSS_SIGMA, self.inspection_image)
        if self.debug:
            show_image(self.inspection_image)

    def _register(self):
        alinger = ImageAligner(self.reference_image, self.inspection_image)
        alinger.find_allignment()
        self.reference_image_registered = alinger.transform(self.reference_image)
        self.valid_registration_mask = alinger.get_valid_mask(self.reference_image.shape)

    def _diff(self):
        # diff_image = relative_diff(reference_image_registered, self.inspection_image)
        diff_image = cv2.absdiff(self.reference_image_registered, self.inspection_image)
        diff_mask = apply_hysteresis_threshold(diff_image, self.LOW_DIFF_THRESHOLD, self.HIGH_DIFF_THRESHOLD)
        valid_diff_mask = np.bitwise_and(diff_mask, self.valid_registration_mask)
        if self.debug:
            show_image(diff_image)
            show_image(valid_diff_mask)
        return valid_diff_mask

    def _post_process(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPHOLOGY_SE_SIZE)
        opening = cv2.morphologyEx(mask.astype(dtype=np.uint8), cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        if self.debug:
            show_image(opening)
            show_image(closing)
        return closing.astype(np.bool)


def main():
    images_dir = 'images'
    debug = True

    for reference_image, inspection_image in DataHandler(images_dir).get():
        detector = DefectDetector(reference_image, inspection_image, debug=debug)
        detector.run()


if __name__ == '__main__':
    main()
