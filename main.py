import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import apply_hysteresis_threshold

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
    MORPHOLOGY_SE_SIZE = (3, 3)

    def __init__(self, reference_image, inspection_image, debug=False):
        self.reference_image = reference_image
        self.inspection_image = inspection_image
        self.debug = debug
        self.reference_image_registered = None
        self.valid_registration_mask = None

    def run(self):
        # self._pre_process()
        self._register()
        valid_diff_mask = self._diff()
        output_mask = self._post_process(valid_diff_mask)
        show_image(output_mask)

        print('valid_diff_mask mean = {}'.format(np.mean(valid_diff_mask.flatten())))
        plt.close('all')
        return

        # Second type of defects:

        # LOW_DIFF_THRESHOLD = 1.3
        # HIGH_DIFF_THRESHOLD = 1.7
        # diff_image = relative_diff(self.reference_image_registered, self.inspection_image)
        # diff_mask = apply_hysteresis_threshold(diff_image, LOW_DIFF_THRESHOLD, HIGH_DIFF_THRESHOLD)
        # valid_diff_mask = np.bitwise_and(diff_mask, self.valid_registration_mask)
        # if self.debug:
        #     show_image(diff_image)
        #     show_image(valid_diff_mask)
        valid_diff_mask = self._diff()

        MORPHOLOGY_SE_SIZE = (5, 5)
        edges = cv2.Canny(self.inspection_image, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPHOLOGY_SE_SIZE)
        cv2.dilate(edges, kernel, edges)
        show_image(edges)

        diff_no_edges = np.logical_and(valid_diff_mask, np.logical_not(edges))
        show_image(diff_no_edges, 'diff no edges')

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
        CROSS_CORELLATION = False
        alinger = ImageAligner(self.reference_image, self.inspection_image)
        if CROSS_CORELLATION:
            from skimage.feature import register_translation
            shift, error, diffphase = register_translation(self.inspection_image, self.reference_image, 1000)#, space="fourier")
            t = TranslationTransform(*shift)
            alinger.translation_model = t
        else:
            alinger.find_allignment()
        self.reference_image_registered = alinger.transform(self.reference_image)
        self.valid_registration_mask = alinger.get_valid_mask(self.reference_image.shape)

        # fine-tune translation (sub-pixel):
        def sliding_window(image, window_size, step_size=1):
            assert divmod(window_size, 2)[1] == 1
            half_window = divmod(window_size, 2)[0]
            # slide a window across the image
            for y in range(half_window, image.shape[0]-half_window):
                for x in range(half_window, image.shape[1]-half_window):
                    # yield the current window
                    yield (x, y, image[y-half_window:y + half_window + 1, x-half_window:x + half_window + 1])

        a1 = np.where(self.valid_registration_mask)
        y_valid_range, x_valid_range = [[a2.min(), a2.max()] for a2 in a1]

        FILTER_SIZE = 5
        A = []
        b = []
        for x, y, window in sliding_window(self.reference_image_registered, FILTER_SIZE):
            A.append(window.flatten())
            b.append(self.inspection_image[y, x])
        x, residuals, rank, s = np.linalg.lstsq(A, b)
        filter = x.reshape(FILTER_SIZE, FILTER_SIZE)
        dst = cv2.filter2D(self.reference_image_registered, -1, filter)
        if self.debug:
            show_image(self.reference_image_registered, 'translation')
            show_image(dst, 'subpixel translation')
        self.reference_image_registered = dst
        pass


    def _diff(self):
        # diff_image = relative_diff(self.reference_image_registered, self.inspection_image)
        diff_image = cv2.absdiff(self.reference_image_registered, self.inspection_image)
        diff_mask = apply_hysteresis_threshold(diff_image, self.LOW_DIFF_THRESHOLD, self.HIGH_DIFF_THRESHOLD)
        valid_diff_mask = np.bitwise_and(diff_mask, self.valid_registration_mask)
        if self.debug:
            show_image(diff_image)
            show_image(valid_diff_mask)
        return valid_diff_mask

    def _post_process(self, mask):
        CONNECTED_COMPONENTS = True
        if CONNECTED_COMPONENTS:
            def connected_components(mask, min_size=30):
                nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
                sizes = stats[1:, -1]  # remove background
                nb_components = nb_components - 1  # remove background

                plt.figure()
                plt.imshow(output, cmap='summer')
                plt.show()

                output_mask = np.zeros(output.shape, dtype=np.bool)
                for i in range(0, nb_components):
                    if sizes[i] >= min_size:
                        output_mask[output == i + 1] = True
                return output_mask
            output = connected_components(mask)

        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPHOLOGY_SE_SIZE)
            opening = cv2.morphologyEx(mask.astype(dtype=np.uint8), cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            output = closing.astype(np.bool)
            if self.debug:
                show_image(opening)
                show_image(closing)
        return output


def main():
    images_dir = 'images'
    debug = True

    for reference_image, inspection_image in DataHandler(images_dir).get():
        detector = DefectDetector(reference_image, inspection_image, debug=debug)
        detector.run()


if __name__ == '__main__':
    main()
