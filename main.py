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
        diff_image = self._diff()
        valid_diff_mask = self._diff_binarization(diff_image)
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
        CROSS_CORELLATION = True
        alinger = ImageAligner(self.reference_image, self.inspection_image)
        if CROSS_CORELLATION:
            from skimage.feature import register_translation
            shift, error, diffphase = register_translation(self.inspection_image, self.reference_image, 100)#, space="fourier")
            t = TranslationTransform(*reversed(shift))
            alinger.translation_model = t
        else:
            alinger.find_allignment()
        self.reference_image_registered = alinger.transform(self.reference_image)
        self.valid_registration_mask = alinger.get_valid_mask(self.reference_image.shape)

        # fine-tune translation (sub-pixel):
        FINE_TUNE = False
        if FINE_TUNE:
            def sliding_window(image, window_size, y_valid_range, x_valid_range):
                dm = divmod(window_size, 2)
                assert dm[1] == 1
                half_window = dm[0]
                # slide a window across the image
                for y in range(max(half_window, y_valid_range[0]), min(image.shape[0]-half_window, y_valid_range[1])):
                    for x in range(max(half_window, x_valid_range[0]), min(image.shape[1]-half_window, x_valid_range[1])):
                        # yield the current window
                        yield (x, y, image[y-half_window:y + half_window + 1, x-half_window:x + half_window + 1])

            a1 = np.where(self.valid_registration_mask)
            y_valid_range, x_valid_range = [[a2.min(), a2.max()] for a2 in a1]

            FILTER_SIZE = 5
            A = []
            b = []
            for x, y, window in sliding_window(self.reference_image_registered, FILTER_SIZE, y_valid_range, x_valid_range):
                A.append(window.flatten())
                b.append(self.inspection_image[y, x])

            SCIPY_LSTSQ = True
            if SCIPY_LSTSQ:
                LOSS = 'soft_l1'#'cauchy'
                from scipy.optimize import least_squares
                def fun(x, A, b):
                    # return x[0] + x[1] * np.exp(x[2] * t) - y
                    return b - np.matmul(A, x)
                x0 = np.zeros((FILTER_SIZE, FILTER_SIZE)).flatten()
                x0[int(x0.size/2)] = 1.0
                res_log = least_squares(fun, x0, loss=LOSS, args=(np.array(A), np.array(b)))
                x = res_log.x
            else:
                x, residuals, rank, s = np.linalg.lstsq(A, b)

            filter = x.reshape(FILTER_SIZE, FILTER_SIZE)
            if self.debug:
                show_image(filter, 'filter')

            dst = cv2.filter2D(self.reference_image_registered, -1, filter)
            if self.debug:
                show_image(self.reference_image_registered, 'translation')
                show_image(dst, 'subpixel translation')
            self.reference_image_registered = dst
            pass

    def _diff(self):
        DIFF = 'abs'#'abs''rel'
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
        DIFF_REGION = True
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

    def _diff_binarization(self, diff_image):
        diff_mask = apply_hysteresis_threshold(diff_image, self.LOW_DIFF_THRESHOLD, self.HIGH_DIFF_THRESHOLD)
        valid_diff_mask = np.bitwise_and(diff_mask, self.valid_registration_mask)
        if self.debug:
            show_image(diff_image)
            show_image(valid_diff_mask)
            print('diff_image mean = {}'.format(np.mean(diff_image.flatten())))
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
