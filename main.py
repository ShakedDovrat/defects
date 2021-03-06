import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.filters import apply_hysteresis_threshold
from skimage.feature import register_translation

from utils import TranslationTransform, show_image, DataHandler


class DefectDetector:
    CANNY_LOW_THRESHOLD = 100
    CANNY_HIGH_THRESHOLD = 200
    EDGES_DILATE_SE_SIZE = (5, 5)
    JOINT_EDGES_FACTOR = 1/3
    LOW_DIFF_THRESHOLD = 20
    HIGH_DIFF_THRESHOLD = 40
    POST_PROCESS_CLOSE_SE_SIZE = (3, 3)
    MIN_CONNECTED_COMPONENT_SIZE = 40

    def __init__(self, reference_image, inspection_image, image_idx, debug=False, output_dir=None):
        self.reference_image = reference_image
        self.inspection_image = inspection_image
        self.image_idx = image_idx
        self.reference_image_registered = None
        self.valid_registration_mask = None
        self.diff_image = None

        self.debug = debug
        self.debug_images = []
        self.output_dir = output_dir
        self.figures_count = 0

    def run(self):
        if self.debug:
            self._save_image(self.inspection_image, 'Input inspection image')
        self._register()
        self._diff()
        joint_edges_mask = self._joint_edges()
        self._lower_diff_at_edges(joint_edges_mask)
        valid_diff_mask = self._diff_binarization()
        output_mask = self._post_process(valid_diff_mask)

        if self.debug:
            show_image(output_mask, 'output_mask')
            self._save_image(output_mask, 'Output: After cleaning')
            plt.close('all')  # Breakpoint location

        return output_mask

    def _register(self):
        shift, _, _ = register_translation(self.inspection_image, self.reference_image, 10)
        tt = TranslationTransform(shift[1], shift[0])
        self.reference_image_registered = tt.transform(self.reference_image)
        self.valid_registration_mask = tt.get_valid_mask(self.reference_image.shape)

    def _diff(self):
        self.diff_image = cv2.absdiff(self.reference_image_registered, self.inspection_image)
        if self.debug:
            show_image(self.diff_image, 'diff_image')
            print('diff_image mean = {}'.format(np.mean(self.diff_image.flatten())))
            self._save_image(np.copy(self.diff_image), 'Diff image after registration with reference')

    def _joint_edges(self):
        inspection_edges = DefectDetector._edges_dilate(self.inspection_image)
        reference_edges = DefectDetector._edges_dilate(self.reference_image_registered)
        joint_edges_mask = np.logical_and(inspection_edges, reference_edges)
        if self.debug:
            show_image(joint_edges_mask, 'joint_edges_mask')
            self._save_image(joint_edges_mask, 'Dilated edges that appear on both images')
        return joint_edges_mask

    @staticmethod
    def _edges_dilate(image):
        edges = cv2.Canny(image, DefectDetector.CANNY_LOW_THRESHOLD, DefectDetector.CANNY_HIGH_THRESHOLD)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DefectDetector.EDGES_DILATE_SE_SIZE)
        cv2.dilate(edges, kernel, edges)
        return edges

    def _lower_diff_at_edges(self, edges_mask):
        self.diff_image[edges_mask] = self.diff_image[edges_mask] * self.JOINT_EDGES_FACTOR
        if self.debug:
            show_image(self.diff_image, 'diff image lower joint edges')
            self._save_image(np.copy(self.diff_image), 'Diff image, lowered at edges')

    def _diff_binarization(self):
        diff_mask = apply_hysteresis_threshold(self.diff_image, self.LOW_DIFF_THRESHOLD, self.HIGH_DIFF_THRESHOLD)
        valid_diff_mask = np.bitwise_and(diff_mask, self.valid_registration_mask)
        if self.debug:
            show_image(valid_diff_mask, 'valid_diff_mask')
            self._save_image(valid_diff_mask, 'Diff mask')
        return valid_diff_mask

    def _post_process(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.POST_PROCESS_CLOSE_SE_SIZE)
        close = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool)
        # open = cv2.morphologyEx(close.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(np.bool)
        if self.debug:
            show_image(close, 'morph close')
        return self._remove_small_connected_components(close)

    def _remove_small_connected_components(self, mask):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        sizes = stats[1:, -1]  # remove background
        nb_components = nb_components - 1  # remove background

        output_mask = np.zeros(output.shape, dtype=np.bool)
        for i in range(0, nb_components):
            if sizes[i] >= self.MIN_CONNECTED_COMPONENT_SIZE:
                output_mask[output == i + 1] = True
        if self.debug:
            show_image(output_mask, 'remove small CCs')
        return output_mask

    def _save_image(self, image, title):
        fig = plt.figure()
        plt.imshow(image)
        plt.title(title)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        file_name = 'image-{}-fig-{}-{}.png'.format(self.image_idx,
                                                    self.figures_count,
                                                    title.lower().replace(' ', '-').replace(',', ''))
        plt.savefig(os.path.join(self.output_dir, file_name))
        plt.close(fig)
        self.figures_count += 1


def main():
    images_dir = 'images'
    output_dir = 'output'
    debug = True

    for image_idx, (reference_image, inspection_image) in enumerate(DataHandler(images_dir).get()):
        detector = DefectDetector(reference_image, inspection_image, image_idx, debug=debug, output_dir=output_dir)
        output_mask = detector.run()
        plt.imsave(os.path.join(output_dir, 'output-image-idx-{}.png'.format(image_idx)), output_mask, cmap=cm.gray)


if __name__ == '__main__':
    main()
