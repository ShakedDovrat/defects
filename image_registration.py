import cv2
import numpy as np
from skimage.measure import ransac
import scipy


class TranslationTransform:
    def __init__(self, temp_dx=None, temp_dy=None):
        self.dx = temp_dx
        self.dy = temp_dy

    def estimate(self, data):
        x1, y1, x2, y2 = data[0]
        self.dx = x2 - x1
        self.dy = y2 - y1
        return True

    def residuals(self, data):
        points1 = np.copy(data[:, :2])
        points2 = np.copy(data[:, 2:])
        points1 += (self.dx, self.dy)
        return [np.linalg.norm(diff, ord=2) for diff in points2 - points1]

    def get_transform_matrix(self):
        translation_matrix = np.eye(3)
        translation_matrix[0, 2] = self.dx
        translation_matrix[1, 2] = self.dy
        return translation_matrix

    def get_transform_params(self):
        return self.dx, self.dy


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


class ImageAligner:
    def __init__(self, image1, image2, debug=False):
        self.image1 = image1
        self.image2 = image2
        self.debug = debug
        self.matches_image = None
        self.translation_model = None

    def find_allignment(self):
        """Code taken from: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
        And modified"""

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(self.image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(self.image2, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        if self.debug:
            self.matches_image = cv2.drawMatches(self.image1, keypoints1, self.image2, keypoints2, matches, None)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find translation
        self.translation_model, inliers = ransac(np.concatenate((points1, points2), axis=1), TranslationTransform,
                                                 min_samples=1, residual_threshold=0.1, max_trials=200)

    def transform(self, image):
        # return cv2.warpPerspective(image, self.translation_model.get_transform_matrix(), list(reversed(image.shape)),
        #                            flags=cv2.INTER_LINEAR)#cv2.INTER_CUBIC)
        return scipy.ndimage.interpolation.shift(image, list(reversed(self.translation_model.get_transform_params())))

    def get_valid_mask(self, image_shape):
        dx, dy = self.translation_model.get_transform_params()
        # dx, dy = np.int(np.round(dx)), np.int(np.round(dy))
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
