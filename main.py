import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import apply_hysteresis_threshold, threshold_otsu

from image_registration import ImageAligner, TranslationTransform


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


def relative_diff(a, b):
    return np.maximum(a, b) / np.minimum(a, b)


def show_image(image, title=''):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.show()


class DefectDetector:
    MEDIAN_FILTER_SIZE = 3
    # LOW_DIFF_THRESHOLD = 1.2
    # HIGH_DIFF_THRESHOLD = 1.7
    LOW_DIFF_THRESHOLD = 70
    HIGH_DIFF_THRESHOLD = 110
    LOW_DIFF_THRESHOLD = 20
    HIGH_DIFF_THRESHOLD = 30
    LOW_DIFF_THRESHOLD = 20
    HIGH_DIFF_THRESHOLD = 40
    # LOW_DIFF_THRESHOLD = 30
    # HIGH_DIFF_THRESHOLD = 60
    MORPHOLOGY_SE_SIZE = (3, 3)

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
        JOINT_EDGES_FACTOR = 1 / 3
        diff_image[joint_edges_mask] = diff_image[joint_edges_mask] * JOINT_EDGES_FACTOR
        if self.debug:
            show_image(diff_image, 'diff image lower joint edges')
        valid_diff_mask = self._diff_binarization(diff_image)
        output_mask = self._post_process(valid_diff_mask)
        if self.debug:
            show_image(output_mask, 'output_mask')

        print('valid_diff_mask mean = {}'.format(np.mean(valid_diff_mask.flatten())))
        plt.close('all')

    def _register(self):
        CROSS_CORELLATION = True
        alinger = ImageAligner(self.reference_image, self.inspection_image)
        if CROSS_CORELLATION:
            from skimage.feature import register_translation
            shift, error, diffphase = register_translation(self.inspection_image, self.reference_image, 10)#, space="fourier")
            t = TranslationTransform(*reversed(shift))
            alinger.translation_model = t
        else:
            alinger.find_allignment()
        self.reference_image_registered = alinger.transform(self.reference_image)
        self.valid_registration_mask = alinger.get_valid_mask(self.reference_image.shape)

    def _diff(self):
        DIFF = 'abs'#''normed'
        if DIFF == 'rel':
            self.LOW_DIFF_THRESHOLD = 1.4#1.2
            self.HIGH_DIFF_THRESHOLD = 1.8#1.7
            diff_image = relative_diff(self.reference_image_registered, self.inspection_image)
            plt.figure()
            plt.imshow(diff_image, vmin=1.0, vmax=5.0)
            plt.show()
        elif DIFF == 'abs':
            diff_image = cv2.absdiff(self.reference_image_registered, self.inspection_image)
        elif DIFF == 'normed':
            self.LOW_DIFF_THRESHOLD = 0.25
            self.HIGH_DIFF_THRESHOLD = 0.6
            diff_image = cv2.absdiff(self.reference_image_registered, self.inspection_image)
            mean_image = np.mean(np.concatenate((np.expand_dims(self.reference_image_registered, axis=2),
                                                 np.expand_dims(self.inspection_image, axis=2)), axis=2), axis=2)
            diff_image = diff_image / mean_image
        else:
            raise ValueError('')
        DIFF_REGION = False
        if DIFF_REGION:
            BLUR_SIZE = 5
            diff_image = cv2.blur(diff_image, (BLUR_SIZE, BLUR_SIZE))
            NORM_AFTER_REGION = True
            if NORM_AFTER_REGION:
                mean_image = np.mean(np.concatenate((np.expand_dims(self.reference_image_registered, axis=2),
                                                     np.expand_dims(self.inspection_image, axis=2)), axis=2), axis=2)
                diff_image = diff_image / mean_image
                plt.figure()
                plt.imshow(diff_image, vmin=1.0, vmax=2.0)
                plt.show()
        if self.debug:
            show_image(diff_image, 'diff_image')
        return diff_image

    def _diff_binarization(self, diff_image):
        OTSU = False
        if OTSU:
            threshold = threshold_otsu(diff_image)
            diff_mask = diff_image > threshold
        else:
            diff_mask = apply_hysteresis_threshold(diff_image, self.LOW_DIFF_THRESHOLD, self.HIGH_DIFF_THRESHOLD)
        valid_diff_mask = np.bitwise_and(diff_mask, self.valid_registration_mask)
        if self.debug:
            # show_image(diff_image, 'diff_image')
            show_image(diff_mask, 'diff_mask')
            # show_image(valid_diff_mask, 'valid_diff_mask')
            print('diff_image mean = {}'.format(np.mean(diff_image.flatten())))
        return valid_diff_mask

    def _joint_edges(self):
        inspection_edges = DefectDetector._edges_dilate(self.inspection_image)
        reference_edges = DefectDetector._edges_dilate(self.reference_image_registered)
        joint_edges_mask = np.logical_and(inspection_edges, reference_edges)
        if self.debug:
        #     show_image(inspection_edges, 'inspection_edges')
        #     show_image(reference_edges, 'reference_edges')
            show_image(joint_edges_mask, 'joint_edges_mask')
        return joint_edges_mask

    @staticmethod
    def _edges_dilate(image):
        MORPHOLOGY_SE_SIZE = (5, 5)

        edges = cv2.Canny(image, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPHOLOGY_SE_SIZE)
        cv2.dilate(edges, kernel, edges)
        return edges

    def _post_process(self, mask):
        CONNECTED_COMPONENTS = True
        if CONNECTED_COMPONENTS:
            def connected_components(mask, min_size=40):
                nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                sizes = stats[1:, -1]  # remove background
                nb_components = nb_components - 1  # remove background

                # plt.figure()
                # plt.imshow(output, cmap='summer')
                # plt.show()

                output_mask = np.zeros(output.shape, dtype=np.bool)
                for i in range(0, nb_components):
                    if sizes[i] >= min_size:
                        output_mask[output == i + 1] = True
                show_image(output_mask, 'remove small CCs')
                return output_mask

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPHOLOGY_SE_SIZE)
            close = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool)
            show_image(close, 'morph close')

            output = connected_components(close)

        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPHOLOGY_SE_SIZE)
            opening = cv2.morphologyEx(mask.astype(dtype=np.uint8), cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            output = closing.astype(np.bool)
            if self.debug:
                show_image(opening)
                show_image(closing)
        # if self.debug:
        #     show_image(output, 'post process')
        return output


def main():
    images_dir = 'images'
    debug = True

    for reference_image, inspection_image in DataHandler(images_dir).get():
        detector = DefectDetector(reference_image, inspection_image, debug=debug)
        detector.run()


if __name__ == '__main__':
    main()
