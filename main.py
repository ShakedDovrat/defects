import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import apply_hysteresis_threshold
from skimage.feature import register_translation

from image_registration import TranslationTransform


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


class DefectDetector:
    LOW_DIFF_THRESHOLD = 20
    HIGH_DIFF_THRESHOLD = 40
    MORPHOLOGY_SE_SIZE = (3, 3)
    EDGES_DILATE_SE_SIZE = (5, 5)
    JOINT_EDGES_FACTOR = 1 / 3

    def __init__(self, reference_image, inspection_image, debug=False):
        self.reference_image = reference_image
        self.inspection_image = inspection_image
        self.debug = debug
        self.reference_image_registered = None
        self.valid_registration_mask = None

    def run(self):
        self._register()
        diff_image = self._diff()
        joint_edges_mask = self._joint_edges()
        diff_image[joint_edges_mask] = diff_image[joint_edges_mask] * self.JOINT_EDGES_FACTOR
        if self.debug:
            show_image(diff_image, 'diff image lower joint edges')
        valid_diff_mask = self._diff_binarization(diff_image)
        output_mask = self._post_process(valid_diff_mask)
        if self.debug:
            show_image(output_mask, 'output_mask')
            plt.close('all')

    def _register(self):
        shift, _, _ = register_translation(self.inspection_image, self.reference_image, 10)
        tt = TranslationTransform(*reversed(shift))
        self.reference_image_registered = tt.transform(self.reference_image)
        self.valid_registration_mask = tt.get_valid_mask(self.reference_image.shape)

    def _diff(self):
        diff_image = cv2.absdiff(self.reference_image_registered, self.inspection_image)
        if self.debug:
            show_image(diff_image, 'diff_image')
            print('diff_image mean = {}'.format(np.mean(diff_image.flatten())))
        return diff_image

    def _diff_binarization(self, diff_image):
        diff_mask = apply_hysteresis_threshold(diff_image, self.LOW_DIFF_THRESHOLD, self.HIGH_DIFF_THRESHOLD)
        valid_diff_mask = np.bitwise_and(diff_mask, self.valid_registration_mask)
        if self.debug:
            show_image(diff_mask, 'diff_mask')
        return valid_diff_mask

    def _joint_edges(self):
        inspection_edges = DefectDetector._edges_dilate(self.inspection_image)
        reference_edges = DefectDetector._edges_dilate(self.reference_image_registered)
        joint_edges_mask = np.logical_and(inspection_edges, reference_edges)
        if self.debug:
            show_image(joint_edges_mask, 'joint_edges_mask')
        return joint_edges_mask

    @staticmethod
    def _edges_dilate(image):
        edges = cv2.Canny(image, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DefectDetector.EDGES_DILATE_SE_SIZE)
        cv2.dilate(edges, kernel, edges)
        return edges

    def _remove_small_connected_components(self, mask, min_size=40):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        sizes = stats[1:, -1]  # remove background
        nb_components = nb_components - 1  # remove background

        output_mask = np.zeros(output.shape, dtype=np.bool)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                output_mask[output == i + 1] = True
        if self.debug:
            show_image(output_mask, 'remove small CCs')
        return output_mask

    def _post_process(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPHOLOGY_SE_SIZE)
        close = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool)
        if self.debug:
            show_image(close, 'morph close')
        return self._remove_small_connected_components(close)


def main():
    images_dir = 'images'
    debug = True

    for reference_image, inspection_image in DataHandler(images_dir).get():
        detector = DefectDetector(reference_image, inspection_image, debug=debug)
        detector.run()


if __name__ == '__main__':
    main()
